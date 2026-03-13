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
        description="Name of the requested dish or recipe, if explicitly present"
    )
    ingredients: List[str] = Field(
        default_factory=list,
        description="List of ingredients explicitly mentioned by the user"
    )


class SearchRecipeInput(BaseModel):
    query: str = Field(description="User instruction to find a recipe")


class ScrapeRecipeInput(BaseModel):
    url: str = Field(description="Recipe page URL to scrape")


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
# Utilitaires
# =========================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def build_search_query(request: RecipeRequest) -> str:
    if request.dish:
        return request.dish

    if request.ingredients:
        return " ".join(request.ingredients)

    return "recipe"


def build_jocooks_search_url(request: RecipeRequest) -> str:
    query = build_search_query(request)
    return f"https://www.jocooks.com/?s={quote_plus(query)}"
def is_jocooks_recipe_url(url: str) -> bool:
    return (
        url.startswith("https://www.jocooks.com/recipes/")
        and url.rstrip("/").count("/") >= 4
    )
def find_first_recipe_url(search_url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(search_url, headers=headers, timeout=15)
    print("status:", response.status_code)
    print("final_url:", response.url)
    print("content-type:", response.headers.get("Content-Type"))
    print("body preview:", response.text[:500])

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Priorité : les cartes de résultats article.entry
    for article in soup.select("article.entry"):
        a = article.select_one("h2.entry-title a[href]")
        if not a:
            continue

        href = a["href"].strip()
        full_url = urljoin("https://www.jocooks.com", href)

        if is_jocooks_recipe_url(full_url):
            return full_url

    # Fallback : tout lien vers /recipes/
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin("https://www.jocooks.com", href)

        if is_jocooks_recipe_url(full_url):
            return full_url

    raise ValueError(f"Aucune URL de recette trouvée depuis {search_url}")

def extract_recipe_request(query: str) -> RecipeRequest:
    prompt = f"""
You analyze a user request about a recipe.

Extract:
- dish: recipe or dish name if explicitly requested
- ingredients: list of explicitly mentioned ingredients

Rules:
- If no precise dish is requested, set dish to null
- If no ingredients are mentioned, return an empty list
- Do not invent anything
- Respond only with valid JSON
- Expected format:
{{
  "dish": null,
  "ingredients": []
}}

User request:
{query}
"""
    response = model.invoke(prompt)

    raw = response.content
    print("[DEBUG extract_recipe_request] type(content) =", type(raw))
    print("[DEBUG extract_recipe_request] raw content =", repr(raw))

    if raw is None:
        raise ValueError("The model returned empty content.")

    if not isinstance(raw, str):
        raw = str(raw)

    raw = raw.strip()
    if not raw:
        raise ValueError("The model returned an empty string.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Non-JSON response from model: {raw}")
        data = json.loads(match.group(0))

    return RecipeRequest(**data)


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def looks_like_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def extract_recipe_json_ld(soup: BeautifulSoup) -> Optional[dict]:
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        candidates = data if isinstance(data, list) else [data]

        for item in candidates:
            if not isinstance(item, dict):
                continue

            if item.get("@type") == "Recipe":
                return item

            graph = item.get("@graph")
            if isinstance(graph, list):
                for sub in graph:
                    if isinstance(sub, dict) and sub.get("@type") == "Recipe":
                        return sub

    return None

# =========================
# Tools
# =========================

@tool(args_schema=SearchRecipeInput)
def search_recipe(query: str) -> str:
    """
    Finds a JoCooks recipe from a user instruction
    and returns the first recipe URL found.
    """
    print(f"[TOOL] search_recipe called with query={query}")

    try:
        request = extract_recipe_request(query)
        print("[DEBUG] request =", request.model_dump())

        search_url = build_jocooks_search_url(request)
        print("[DEBUG] search_url =", search_url)

        recipe_url = find_first_recipe_url(search_url)
        print("[DEBUG] recipe_url =", recipe_url)

        return recipe_url

    except Exception as e:
        print("[ERROR search_recipe]", repr(e))
        return f"Error while searching recipe: {str(e)}"


def normalize_instruction_text(text: str) -> str:
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_instruction_lines(instructions) -> list[str]:
    lines = []

    def walk(node):
        if not node:
            return

        if isinstance(node, str):
            text = normalize_instruction_text(node)
            if text:
                lines.append(text)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)
            return

        if isinstance(node, dict):
            node_type = node.get("@type", "")
            
            # Cas direct: {"@type": "HowToStep", "text": "..."}
            text = node.get("text")
            if isinstance(text, str):
                clean = normalize_instruction_text(text)
                if clean:
                    lines.append(clean)

            # Cas section: {"@type": "HowToSection", "itemListElement": [...]}
            item_list = node.get("itemListElement")
            if item_list:
                walk(item_list)

            # Fallback sur d'autres champs parfois utilisés
            for key in ("name", "description"):
                value = node.get(key)
                if isinstance(value, str):
                    clean = normalize_instruction_text(value)
                    if clean and clean not in lines:
                        lines.append(clean)

    walk(instructions)

    # dédoublonnage en gardant l'ordre
    deduped = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    return deduped


@tool(args_schema=ScrapeRecipeInput)
def scrape_recipe(url: str) -> str:
    """
    Downloads a recipe page and extracts structured recipe data when possible.
    Falls back to generic HTML text extraction otherwise.
    """
    print(f"[TOOL] scrape_recipe called with url={url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return f"Error: content at {url} is not an HTML page."

        soup = BeautifulSoup(response.text, "html.parser")

        # 1) Priorité au JSON-LD Recipe
        recipe_data = extract_recipe_json_ld(soup)
        if recipe_data:
            title = recipe_data.get("name", "Unknown title")
            ingredients = recipe_data.get("recipeIngredient", []) or []
            instructions = recipe_data.get("recipeInstructions", [])
            instruction_lines = extract_instruction_lines(instructions)

            parts = [
                f"Title: {title}",
                f"Source: {url}",
            ]

            if ingredients:
                parts.append("\nIngredients:")
                parts.extend(f"- {ing}" for ing in ingredients[:20])

            if instruction_lines:
                parts.append("\nInstructions:")
                parts.extend(
                    f"{i+1}. {step}" for i, step in enumerate(instruction_lines[:10])
                )

            return "\n".join(parts)

        # 2) Fallback HTML générique
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = "Unknown title"
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
            content = content[:3000] + "\n\n[text truncated]"

        return f"Title: {title}\nSource: {url}\n\nExtracted content:\n{content}"

    except requests.RequestException as e:
        return f"Error while scraping {url}: {str(e)}"


# =========================
# Agent
# =========================
agent = create_agent(
    model=model,
    tools=[search_recipe, scrape_recipe],
    system_prompt=(
        "You are an agent that helps find and display an English recipe from Jo Cooks.\n"
        "Important rules:\n"
        "1. If the user gives a cooking request or a list of ingredients, use search_recipe first.\n"
        "2. If you get a recipe URL, then use scrape_recipe.\n"
        "3. If the user directly provides a URL, directly use scrape_recipe.\n"
        "4. At the end, display cleanly:\n"
        "   - the title if available\n"
        "   - the source\n"
        "   - ingredients if available\n"
        "   - the full recipe or all the instructions provided \n"
        "5. Do not invent any information.\n"
    ),
)


# =========================
# CLI
# =========================

if __name__ == "__main__":
    print("=== Recipe CLI Agent ===")
    print("Example prompts:")
    print("- I want a cheesecake recipe")
    print("- What can I cook with eggs, butter and flour?")
    print("- https://www.allrecipes.com/recipe/...")
    print()

    user_query = input("Your prompt: ").strip()

    if not user_query:
        print("No prompt provided.")
        raise SystemExit(1)

    try:
        response = agent.invoke({
            "messages": [("human", user_query)]
        })

        print("\n=== AGENT TRACE ===\n")
        for msg in response["messages"]:
            print(type(msg).__name__, "=>", getattr(msg, "content", msg))

        print("\n=== FINAL RESPONSE ===\n")
        print(response["messages"][-1].content)

    except Exception as e:
        print(f"\nError: {str(e)}")