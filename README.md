# project-LLM

## TODO

### A faire
- Avoir un jeu de poker fonctionnel à partir du code de [Pokerkit](https://github.com/uoftcprg/pokerkit)
- Envoyer les données de la partie en cours à un autre service qu'on va créer 
- Avoir un frontend qui s'ajoute au jeu  
- FunctionCalling qui prend en entrée un json contenant l'état actuel du jeu et renvoie ce qu'on lui demande. En première étape, seulement le meilleur coup et ensuite, on pourra personnaliser.
- Plusieurs modes d'aides (prudent, standard, risqué)
- Avoir l'outil qui permet d'avoir la/les probabilités d'une main.
- Stocker les données d'une partie dans un JSON

### Pour aller plus loin
- Proposer une mise en fonction du comportement
- Detection de comportements adverses et comment y réagir

### Evaluation
- Range de proba -> coups possibles/acceptés
- Ajoute une règle :

    Si p(win)>mise/pot+mise -> Call/Bet autorisé
    Sinon -> Fold autorisé

    Taux de décisions EV-positives
    On compte :
        % de fois où le LLM choisit une action avec EV > 0
        % de fois où il choisit une action EV < 0
    
- Calculer l'EV de chaque coup (bet, call, fold), prendre l'EV optimal et prendre l'EV du LLM et calculer un regret = EV(max) - EV(LLM)
- Faire X parties et calculer la moyenne des stats 

## Répartition

Personne 1 : PokerKit/Pypoker et Chercher frontend, connecter tout ça, pouvoir faire une partie + envoyer les données 
Personne 2 : Préparer tous les modéles de question et de réponses à l'IA. Prompt Engineering. Regarder d'autres framework mieux que LangChain.
Personne 3 : En entrée une main et l'état de la game et en sortie le pourcentage de gagner avec ça