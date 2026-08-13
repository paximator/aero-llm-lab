# Présentation de la review de développement

Objectif : présenter en 10 à 12 minutes un projet techniquement crédible, sans
transformer les résultats de développement en promesses de production.

## Message central

> J’ai construit une chaîne locale et reproductible pour répondre à des questions
> sur des rapports aéronautiques : ingestion vérifiable, retrieval, génération,
> QLoRA, évaluation fail-closed et serving typé. Le résultat le plus intéressant
> n’est pas seulement une amélioration mesurée ; c’est aussi la capacité du projet à
> détecter un échec SFT+RAG, à en isoler les causes et à arrêter les expériences qui
> ne franchissent pas leurs portes de qualité.

## Déroulé conseillé

### 0:00–1:00 — Problème et parti pris

À dire :

> Le risque principal d’un RAG sur des rapports de sécurité n’est pas seulement de
> donner une mauvaise réponse. C’est de produire une réponse plausible sans preuve
> traçable. J’ai donc privilégié une architecture evaluation-first : sources
> immuables, splits par famille d’événements, citations exactes et comportement
> fail-closed.

Montrer : `docs/development-review.md`, section “What is demonstrably working”.

### 1:00–3:00 — Chaîne technique

Présenter cette séquence :

```text
NTSB snapshots → PDF parsing → deterministic chunks → BM25/dense/hybrid/reranker
→ grounded generation → strict postprocessing → evaluation → FastAPI serving
```

Points à souligner :

- les documents et chunks sont identifiés par leur contenu ;
- les familles d’événements ne traversent pas les splits ;
- modèles, retrievers et postprocessor sont derrière des frontières injectables ;
- aucun poids de modèle n’est chargé à l’import ;
- le serving de démonstration utilise des fakes déterministes et ne prétend pas être
  un benchmark de production.

### 3:00–5:00 — Résultats positifs

Afficher `docs/results/README.md` et annoncer seulement les chiffres suivants :

1. Retrieval historique : Recall@10 BM25 `0,500`, hybride `0,700`.
2. Reranking : MRR@10 `0,483` sur le test historique verrouillé.
3. Faisabilité locale : Ministral sur RTX 4070 Laptop, pic `4,96 GB`, débit chaud
   `1,12 token/s`.
4. QLoRA gold-context sur 27 questions de développement : accuracy `18,5 % →
   44,4 %`, sorties valides et grounded `0 % → 37,0 %`.

À dire immédiatement après :

> Le résultat QLoRA isole la génération avec un contexte gold. Ce n’est pas une
> preuve de qualité RAG end-to-end.

### 5:00–7:00 — Résultat négatif et démarche scientifique

À dire :

> En contexte réellement retrieved, le hit@3 est de 63 %, mais RAG et SFT+RAG
> échouent fermés sur les 27 exemples. L’analyse montre deux problèmes différents :
> Base produit du texte non JSON ; SFT apprend le préfixe JSON mais tronque sa
> sortie. Un entraînement micro avec EOS supprime la troncature, sans résoudre la
> copie exacte des citations. Le gate reste à 0/5, donc je n’ai pas lancé un run plus
> coûteux.

Ce que cela démontre : instrumentation, taxonomie d’échecs, sélection honnête du
modèle et maîtrise du budget expérimental.

### 7:00–8:30 — Démo serving

Avant la présentation :

```powershell
uv sync --locked --extra dev --extra serving
```

Terminal 1 :

```powershell
uv run --locked --extra serving uvicorn aerollm.serving.app:create_app `
  --factory --host 127.0.0.1 --port 8000
```

Terminal 2 :

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/answer -Method Post `
  -ContentType application/json -Body '{"question":"What happened?"}'
```

Commenter dans la réponse : `request_id`, `latency_ms`, `backend`, sources et format
typé. Préciser que `deterministic-fake` est volontaire pour une démo CI stable.

### 8:30–10:00 — État de l’évaluation et suite

À dire :

> Evaluation-suite v2 contient 27 questions de développement revues. Les 45 slots
> de test sont structurés mais leurs labels doivent encore être rédigés et revus par
> des humains indépendants. Le statut `human_review_and_test_authoring_required` est
> donc volontaire et exact.

Afficher `docs/action-register.md` : A1 et A3 peuvent commencer ; A2, A4 et A5 sont
bloqués ou gated. Terminer par :

> La prochaine limite n’est plus l’infrastructure. C’est la qualité des annotations
> de test et des exemples d’entraînement retrieved-context.

## Plan B si la démo ne démarre pas

Ne pas improviser avec le modèle GPU. Montrer :

1. le run CI vert sur Windows/Linux et Python 3.11/3.13 ;
2. `tests/serving/test_app.py`, notamment health, request ID et réponse déterministe ;
3. les commandes et la sortie vérifiée consignées dans `development-review.md`.

Dire :

> La démo live est un confort. La preuve reproductible reste la CI et les tests.

## Questions difficiles et réponses courtes

**Pourquoi seulement 27 questions revues ?**

Le temps disponible a servi à valider tout le workflow sur le développement. Les
45 slots restants rendent le travail visible, mais leurs labels de test ne seront
pas générés automatiquement : ils exigent auteur et reviewer indépendants.

**Pourquoi publier des résultats négatifs ?**

Ils empêchent une fausse conclusion. Le gold-context montre un gain de génération ;
le test RAG montre que ce gain ne survit pas encore au retrieval réel.

**Pourquoi rejeter des citations presque identiques ?**

Le contrat exige une preuve exacte. Accepter silencieusement une paraphrase ou une
ponctuation inventée transformerait une citation en simple affirmation du modèle.

**Pourquoi ne pas utiliser vLLM ?**

Le besoin actuel est la qualité et la traçabilité, pas le débit multi-requêtes. Le
repo expose une frontière d’adapter locale sans revendiquer une intégration non
implémentée.

**Est-ce production-ready ?**

Non. Le serving est un prototype propre et testé. La suite de test v2, la qualité
SFT+RAG et le benchmark de production ne satisfont pas encore leurs gates.

**Quelle est la contribution la plus mature ?**

La cohérence entre données immuables, prévention de leakage, évaluation séparée,
postprocessing fail-closed, expériences digestées et décisions fondées sur des
portes mesurées.

## Checklist juste avant la review

- [ ] `dev` est propre et synchronisée avec GitHub.
- [ ] Le dernier run CI est vert.
- [ ] Les deux terminaux de démo sont ouverts à la racine du repo.
- [ ] Aucun secret, workbook privé ou chemin personnel n’est affiché.
- [ ] `development-review.md`, `results/README.md` et `action-register.md` sont
  ouverts dans des onglets séparés.
- [ ] Les limitations sont annoncées avant les questions, pas découvertes pendant.
