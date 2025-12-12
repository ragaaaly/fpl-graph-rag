# app.py – full, self-contained Streamlit UI + real FPL back-end
# ============================================================================
import numpy as np
import streamlit as st
import json
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
from pyvis.network import Network
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import atexit
import os

# ------------------------------------------------------
# 0.  Config loader
# ------------------------------------------------------
def read_config(config_file='config.txt'):
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    config['URI'] = config.get('NEO4J_URI') or config.get('URI')
    config['USERNAME'] = config.get('NEO4J_USERNAME') or config.get('USERNAME')
    config['PASSWORD'] = config.get('NEO4J_PASSWORD') or config.get('PASSWORD')
    return config

# ------------------------------------------------------
# 1.  Pre-processing classes (minimal notebook copy)
# ------------------------------------------------------
@dataclass
class ProcessedInput:
    raw_query: str
    intent: str
    confidence: float
    entities: Dict[str, List[str]]
    embedding: Optional[np.ndarray]
    cypher_params: Dict[str, any]

class FPLInputPreprocessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2",
                 team_names_from_kg: Optional[List[str]] = None,
                 player_names_from_kg: Optional[List[str]] = None):
        self.intent_patterns = {
            'player_performance': [r'\b(goals?|assists?|points?|stats?|performance|how many|how much)\b'],
            'player_comparison':  [r'\b(compare|versus|vs\.?|better|who.*better)\b'],
            'team_analysis':      [r'\b(team|defence|attack|clean sheets)\b'],
            'fixture_query':      [r'\b(fixture|gameweek|gw|when.*play)\b'],
            'position_analysis':  [r'\b(top|best).*?\b(defenders?|midfielders?|forwards?|goalkeepers?|def|mid|fwd|gk)\b'],
            'recommendation':     [r'\b(recommend|suggest|should i|captain)\b'],
            'search_player':      [r'\b(tell me about|who is|info about)\b'],
        }
        team_regex = '|'.join(re.escape(t) for t in (team_names_from_kg or [])) or \
                     "arsenal|liverpool|man city|manchester city|chelsea|tottenham|man utd|manchester united|newcastle|brighton|aston villa|west ham|crystal palace|wolves|fulham|brentford|nottm forest|everton|leicester|leeds|southampton|bournemouth|burnley"
        player_regex = '|'.join(re.escape(p) for p in (player_names_from_kg or [])) or \
                      "mohamed salah|harry kane|kevin de bruyne|bruno fernandes|son heung-min|erling haaland"
        self.entity_patterns = {
            'player_name': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
            'team_name': rf'\b({team_regex})\b',
            'position': r'\b(DEF|MID|FWD|GK|defender|midfielder|forward|goalkeeper)\b',
            'season': r'\b(2021-22|2022-23|2023-24)\b',
            'gameweek': r'\b(?:GW|gameweek|week)\s*(\d+)\b',
            'stat_metric': r'\b(goals|assists|points|minutes|clean sheets)\b',
        }
        self.position_map = {'defender':'DEF','midfielder':'MID','forward':'FWD','goalkeeper':'GK',
                             'def':'DEF','mid':'MID','fwd':'FWD','gk':'GK'}
        self.embedding_model = SentenceTransformer(embedding_model_name, device='cpu')

    def classify_intent(self, query: str) -> Tuple[str, float]:
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in self.intent_patterns}
        for intent, patterns in self.intent_patterns.items():
            for pat in patterns:
                scores[intent] += len(re.findall(pat, query_lower))
        if max(scores.values()) == 0: return 'general_query', 0.5
        best = max(scores, key=scores.get)
        return best, min(1.0, scores[best] / len(self.intent_patterns[best]))

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {}
        for ent_type, pat in self.entity_patterns.items():
            matches = re.findall(pat, query, re.IGNORECASE)
            if ent_type == 'gameweek':
                entities[ent_type] = [m if isinstance(m, str) else m[1] for m in matches]
            elif ent_type == 'position':
                entities[ent_type] = list({self.position_map.get(m.lower(), m.upper()) for m in matches})
            else:
                entities[ent_type] = list({m if isinstance(m, str) else m[0] for m in matches})
        if 'player_name' not in entities:
            caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
            entities['player_name'] = [c for c in caps if c.lower() not in {'which','recommend','compare','show'}]
        return {k: v for k, v in entities.items() if v}

    def generate_embedding(self, query: str) -> np.ndarray:
        return self.embedding_model.encode(query, convert_to_numpy=True)

    def build_cypher_params(self, entities: Dict[str, List[str]], intent: str) -> Dict[str, any]:
        p = {}
        if 'player_name' in entities:
            p['player_names'] = entities['player_name']
            p['player_name']  = entities['player_name'][0]
            if intent == 'player_comparison' and len(entities['player_name']) >= 2:
                p['player1'] = entities['player_name'][0]
                p['player2'] = entities['player_name'][1]
        if 'team_name' in entities:
            p['team_names'] = entities['team_name']
            p['team_name']  = entities['team_name'][0]
        if 'position' in entities:
            p['positions'] = entities['position']
            p['position']   = entities['position'][0]
        if 'season' in entities:
            p['seasons'] = entities['season']
            p['season']    = entities['season'][0]
        if 'gameweek' in entities:
            p['gameweeks'] = [int(g) for g in entities['gameweek']]
            p['gameweek']  = int(entities['gameweek'][0])
        return p

    def process(self, query: str) -> ProcessedInput:
        intent, conf = self.classify_intent(query)
        entities = self.extract_entities(query)
        emb = self.generate_embedding(query)
        params = self.build_cypher_params(entities, intent)
        return ProcessedInput(query, intent, conf, entities, emb, params)

# ------------------------------------------------------
# 2.  Cypher templates (≥10)
# ------------------------------------------------------
CYPHER_TEMPLATES = {
    'player_performance': """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
        RETURN  p.player_name AS player,
                SUM(r.total_points) AS total_points,
                SUM(r.goals_scored) AS goals,
                SUM(r.assists) AS assists,
                SUM(r.minutes) AS minutes,
                COUNT(f) AS matches
    """,
    'player_comparison': """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE p.player_name IN [$player1, $player2]
        RETURN  p.player_name AS player,
                SUM(r.total_points) AS points,
                SUM(r.goals_scored) AS goals,
                SUM(r.assists) AS assists,
                SUM(r.minutes) AS minutes
        ORDER BY points DESC
    """,
    'team_analysis': """
        MATCH (t:Team {name: $team_name})<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
        WITH f
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        RETURN  COUNT(DISTINCT f) AS matches,
                SUM(r.goals_scored) AS goals_scored,
                SUM(r.clean_sheets) AS clean_sheets
    """,
    'fixture_query': """
        MATCH (f:Fixture)-[:HAS_HOME_TEAM]->(t1:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(t2:Team)
        WHERE f.gameweek = $gameweek
        RETURN  t1.name AS home, t2.name AS away, f.gameweek AS gameweek
    """,
    'position_analysis': """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, SUM(r.total_points) AS tp, SUM(r.goals_scored) AS g, SUM(r.assists) AS a
        ORDER BY tp DESC
        LIMIT 10
        RETURN  p.player_name AS player, tp AS total_points, g AS goals, a AS assists
    """,
    'season_comparison': """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
        RETURN  f.season AS season,
                SUM(r.total_points) AS points,
                SUM(r.goals_scored) AS goals,
                SUM(r.assists) AS assists
        ORDER BY season
    """,
    'recommendation': """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, SUM(r.total_points) AS tp, AVG(r.form) AS af
        ORDER BY tp DESC
        LIMIT 5
        RETURN  p.player_name AS player, tp AS total_points, af AS avg_form
    """,
    'search_player': """
        MATCH (p:Player {player_name: $player_name})
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(t:Team)
        RETURN  p.player_name AS player, pos.name AS position, t.name AS team
    """,
    'general_query': """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, SUM(r.total_points) AS tp
        ORDER BY tp DESC
        LIMIT 10
        RETURN  p.player_name AS player, tp AS total_points
    """,
}

# ------------------------------------------------------
# 3.  Neo4j wrapper
# ------------------------------------------------------
class Neo4jExecutor:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
    def close(self): self.driver.close()
    def run(self, query, params=None):
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, params or {})]

# ------------------------------------------------------
# 4.  Global singletons
# ------------------------------------------------------
CONFIG   = read_config()
NEO      = Neo4jExecutor(CONFIG['URI'], CONFIG['USERNAME'], CONFIG['PASSWORD'])
team_names   = [r['name'] for r in NEO.run("MATCH (t:Team) RETURN t.name AS name")]
player_names = [r['name'] for r in NEO.run("MATCH (p:Player) RETURN p.player_name AS name")]
PREPROC = FPLInputPreprocessor(team_names_from_kg=team_names, player_names_from_kg=player_names)
atexit.register(NEO.close)

# ------------------------------------------------------
# 5.  Functions the UI calls
# ------------------------------------------------------
def classify_intent(query: str):          
    return PREPROC.classify_intent(query)

def extract_entities(query: str):         
    return PREPROC.extract_entities(query)

def build_cypher(intent, entities):       
    processed = PREPROC.process("dummy")
    processed.entities = entities
    processed.intent   = intent
    params = PREPROC.build_cypher_params(entities, intent)
    tmpl   = CYPHER_TEMPLATES.get(intent, CYPHER_TEMPLATES['general_query'])
    return tmpl, params

def kg_retrieve_baseline(cypher: str, params: dict):
    return NEO.run(cypher, params)

def kg_retrieve_embedding(query_vec, top_k=5):
    # placeholder – swap in real vector search later
    return []

def llm_answer(model: str, context: str, query: str):
    persona = "You are an FPL expert assistant."
    task    = "Answer the question using only the provided knowledge-graph context."
    prompt  = f"{persona}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
    # stub answers – wire real LLM here
    if "gpt-4" in model.lower():
        return f"(GPT-4) Based on the data: {context}"
    if "llama" in model.lower():
        return f"(Llama-3) Based on the data: {context}"
    return f"(GPT-3.5) Based on the data: {context}"

# ============================================================================
# STREAMLIT UI PART (unchanged from earlier version)
# ============================================================================
st.set_page_config(page_title="Graph-RAG FPL Assistant", layout="wide")
SESSION = st.session_state
if "history" not in SESSION:
    SESSION.history = []

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Controls")
    model = st.selectbox("LLM model", ["gpt-3.5", "gpt-4", "llama-3"])
    retrieval = st.radio("Retrieval method", ["Baseline (Cypher)", "Embedding", "Both"], index=0)
    show_cypher = st.checkbox("Show executed Cypher", value=True)
    show_graph = st.checkbox("Show graph snippet", value=True)
    if st.button("🗑️ Clear history"):
        SESSION.history.clear()

# ---------- MAIN ----------
st.title("Graph-RAG FPL Assistant")
st.markdown("Ask anything about players, teams, gameweeks or stats.")

query = st.text_input("Your question:", placeholder="e.g. Best midfielders under 11.0 for 2023")

if st.button("Run") and query:
    with st.spinner("Thinking…"):
        intent, conf = classify_intent(query)
        entities = extract_entities(query)
        cypher_sql, cypher_params = build_cypher(intent, entities)

        kg_context = []
        if retrieval in ["Baseline (Cypher)", "Both"]:
            kg_context += kg_retrieve_baseline(cypher_sql, cypher_params)
        if retrieval in ["Embedding", "Both"]:
            kg_context += kg_retrieve_embedding(PREPROC.generate_embedding(query))
        kg_context = pd.DataFrame(kg_context).drop_duplicates().to_dict("records")

        context_str = json.dumps(kg_context, indent=2, ensure_ascii=False)
        answer = llm_answer(model, context_str, query)

        SESSION.history.append({
            "q": query,
            "cypher": cypher_sql,
            "params": cypher_params,
            "kg": kg_context,
            "answer": answer,
            "model": model,
            "retrieval": retrieval,
            "time": pd.Timestamp.now().strftime("%H:%M:%S")
        })

# ---------- HISTORY ----------
for turn in reversed(SESSION.history):
    st.markdown(f"### 🧑 {turn['q']}  *({turn['time']})*")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Retrieved KG context**")
        st.dataframe(turn["kg"])
    with c2:
        st.markdown("**FPL advice**")
        st.markdown(turn["answer"])

    if show_cypher:
        with st.expander("Cypher query"):
            st.code(turn["cypher"], language="cypher")
            st.json(turn["params"])
    if show_graph and turn["kg"]:
        with st.expander("Graph snippet"):
            import networkx as nx
            G = nx.Graph()
            for rec in turn["kg"]:
                node_id = rec.get('player') or rec.get('player_name') or str(hash(str(rec)))
                G.add_node(node_id, label=node_id, title=str(rec))
            net = Network(height="300px", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            net.repulsion()
            net.show("kg.html")
            HtmlFile = open("kg.html", "r", encoding="utf-8")
            components.html(HtmlFile.read(), height=350)
    st.markdown("---")