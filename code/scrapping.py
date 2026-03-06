import os
import re
import json
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import quote_plus, urljoin

load_dotenv()


# =========================
# Modèles de données
# =========================

class RecipeRequest(BaseModel):
    dish: Optional[str] = Field(
        default=None,
        description="Nom de la recette ou du plat demandé, si présent"
    )
    ingredients: List[str] = Field(
        default_factory=list,
        description="Liste d'ingrédients explicitement mentionnés par l'utilisateur"
    )


class SearchRecipeInput(BaseModel):
    query: str = Field(description="Consigne utilisateur pour trouver une recette")


class ScrapeRecipeInput(BaseModel):
    url: str = Field(description="URL de la page recette à scraper")


# =========================
# LLM
# =========================

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0
)


# =========================
# Fonctions utilitaires
# =========================

def build_search_query(request: RecipeRequest) -> str:
    if request.dish:
        return request.dish

    if request.ingredients:
        return " ".join(request.ingredients)

    return "recette"


def build_marmiton_search_url(request: RecipeRequest) -> str:
    query = build_search_query(request)
    return f"https://www.marmiton.org/recettes/recherche.aspx?aqt={quote_plus(query)}"


def find_first_recipe_url(search_url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    response = requests.get(search_url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/recettes/recette_" in href and href.endswith(".aspx"):
            return urljoin("https://www.marmiton.org", href)

    raise ValueError(f"Aucune URL de recette trouvée depuis {search_url}")


def extract_recipe_request(query: str) -> RecipeRequest:
    prompt = f"""
Tu analyses une demande utilisateur à propos d'une recette.

Extrait :
- dish : le nom du plat ou de la recette si l'utilisateur en demande une explicitement
- ingredients : la liste des ingrédients mentionnés explicitement

Règles :
- Si aucun plat précis n'est demandé, mets dish à null
- Si aucun ingrédient n'est mentionné, retourne une liste vide
- N'invente rien
- Réponds uniquement en JSON valide
- Format attendu :
{{
  "dish": null,
  "ingredients": []
}}

Requête utilisateur :
{query}
"""
    response = model.invoke(prompt)

    raw = response.content
    print("[DEBUG extract_recipe_request] type(content) =", type(raw))
    print("[DEBUG extract_recipe_request] raw content =", repr(raw))

    if raw is None:
        raise ValueError("Le modèle a renvoyé un contenu vide.")

    if not isinstance(raw, str):
        raw = str(raw)

    raw = raw.strip()
    if not raw:
        raise ValueError("Le modèle a renvoyé une chaîne vide.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Réponse non JSON du modèle: {raw}")
        data = json.loads(match.group(0))

    return RecipeRequest(**data)


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def looks_like_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


# =========================
# Tools
# =========================

@tool(args_schema=SearchRecipeInput)
def search_recipe(query: str) -> str:
    """
    Trouve une recette Marmiton à partir d'une consigne utilisateur
    et retourne la première URL de recette trouvée.
    """
    print(f"[TOOL] search_recipe appelé avec query={query}")

    try:
        request = extract_recipe_request(query)
        print("[DEBUG] request =", request.model_dump())

        search_url = build_marmiton_search_url(request)
        print("[DEBUG] search_url =", search_url)

        recipe_url = find_first_recipe_url(search_url)
        print("[DEBUG] recipe_url =", recipe_url)

        return recipe_url

    except Exception as e:
        print("[ERREUR search_recipe]", repr(e))
        return f"Erreur pendant la recherche de recette: {str(e)}"


@tool(args_schema=ScrapeRecipeInput)
def scrape_recipe(url: str) -> str:
    """
    Télécharge une page web et extrait son texte principal.
    Retourne un texte simple avec le titre et le contenu trouvé.
    """
    print(f"[TOOL] scrape_recipe appelé avec url={url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return f"Erreur: le contenu à {url} n'est pas une page HTML."

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = "Titre inconnu"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        chunks = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            text = el.get_text(" ", strip=True)
            if text and len(text) > 2:
                chunks.append(text)

        content = "\n".join(chunks)
        content = clean_text(content)

        if len(content) > 3000:
            content = content[:3000] + "\n\n[texte tronqué]"

        return f"Titre: {title}\nSource: {url}\n\nContenu extrait:\n{content}"

    except requests.RequestException as e:
        return f"Erreur pendant le scraping de {url}: {str(e)}"


# =========================
# Agent
# =========================

agent = create_agent(
    model=model,
    tools=[search_recipe, scrape_recipe],
    system_prompt=(
        "Tu es un agent qui aide à trouver et afficher une recette.\n"
        "Règles importantes :\n"
        "1. Si l'utilisateur donne une consigne culinaire ou une liste d'ingrédients, "
        "utilise d'abord l'outil search_recipe.\n"
        "2. Si tu obtiens une URL de recette, utilise ensuite obligatoirement scrape_recipe.\n"
        "3. Si l'utilisateur fournit directement une URL, utilise directement scrape_recipe.\n"
        "4. À la fin, affiche proprement :\n"
        "   - le titre si disponible\n"
        "   - la source\n"
        "   - un extrait tronqué de la recette\n"
        "5. N'invente aucune information.\n"
    ),
)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    print("=== Agent recette CLI ===")
    print("Tape une consigne, par exemple :")
    print("- Je veux une recette de cheesecake")
    print("- Que faire avec des œufs, du beurre et de la farine ?")
    print("- https://www.marmiton.org/recettes/recette_....aspx")
    print()

    user_query = input("Votre consigne : ").strip()

    if not user_query:
        print("Aucune consigne fournie.")
        raise SystemExit(1)

    try:
        response = agent.invoke({
            "messages": [("human", user_query)]
        })

        print("\n=== TRACE AGENT ===\n")
        for msg in response["messages"]:
            print(type(msg).__name__, "=>", getattr(msg, "content", msg))

        print("\n=== RÉPONSE FINALE ===\n")
        print(response["messages"][-1].content)

    except Exception as e:
        print(f"\nErreur: {str(e)}")