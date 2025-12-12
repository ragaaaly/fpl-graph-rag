# app.py – Final Version: Readable Answers + Robust Schema Detection
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
from pyvis.network import Network
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import atexit
import os

# ------------------------------------------------------
# 0.  Config loader
# ------------------------------------------------------
def read_config(config_file='config.txt'):
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line: continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    config['URI'] = config.get('NEO4J_URI') or config.get('URI') or "bolt://localhost:7687"
    config['USERNAME'] = config.get('NEO4J_USERNAME') or config.get('USERNAME') or "neo4j"
    config['PASSWORD'] = config.get('NEO4J_PASSWORD') or config.get('PASSWORD') or "password"
    return config

# ------------------------------------------------------
# 1.  Neo4j Wrapper & Schema Detection
# ------------------------------------------------------
class Neo4jExecutor:
    def __init__(self, uri, user, pwd):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self.driver.verify_connectivity()
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self): 
        if self.driver: self.driver.close()
        
    def run(self, query, params=None):
        if not self.driver: return []
        try:
            with self.driver.session() as s:
                return [r.data() for r in s.run(query, params or {})]
        except Exception as e:
            return [{"error": str(e)}]

CONFIG = read_config()
NEO = Neo4jExecutor(CONFIG['URI'], CONFIG['USERNAME'], CONFIG['PASSWORD'])
atexit.register(NEO.close)

# --- Schema Auto-Detection ---
@st.cache_resource
def detect_schema_and_data():
    player_prop = "player_name"
    team_prop = "team_name"
    players = []
    teams = []
    
    if NEO.driver:
        try:
            # Detect Player Property
            check1 = NEO.run("MATCH (p:Player) RETURN p.player_name AS name LIMIT 1")
            if not check1 or 'name' not in check1[0] or check1[0]['name'] is None:
                check2 = NEO.run("MATCH (p:Player) RETURN p.name AS name LIMIT 1")
                if check2 and 'name' in check2[0]: player_prop = "name"
            
            # Detect Team Property
            check_t1 = NEO.run("MATCH (t:Team) RETURN t.team_name AS name LIMIT 1")
            if not check_t1 or 'name' not in check_t1[0] or check_t1[0]['name'] is None:
                check_t2 = NEO.run("MATCH (t:Team) RETURN t.name AS name LIMIT 1")
                if check_t2 and 'name' in check_t2[0]: team_prop = "name"

            # Load Data
            p_data = NEO.run(f"MATCH (p:Player) RETURN p.{player_prop} AS name")
            players = [r['name'] for r in p_data if r['name']]
            t_data = NEO.run(f"MATCH (t:Team) RETURN t.{team_prop} AS name")
            teams = [r['name'] for r in t_data if r['name']]
            
        except Exception as e: print(f"Schema Error: {e}")
            
    return player_prop, team_prop, teams, players

PLAYER_PROP, TEAM_PROP, KG_TEAMS, KG_PLAYERS = detect_schema_and_data()

# ------------------------------------------------------
# 2.  Robust Pre-processing
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
    def __init__(self, kg_teams: List[str], kg_players: List[str], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.valid_teams = kg_teams
        self.valid_players = kg_players
        
        self.intent_patterns = {
            'player_performance': [(r'\b(goals?|assists?|points?|stats?|performance|how many|score)\b', 1.8)],
            'player_comparison': [(r'\b(compare|versus|vs\.?|better|who.*better)\b', 2.5)],
            'team_analysis': [(r'\b(team|club|stats?|defence|attack|clean sheets)\b', 2.0)],
            'fixture_query': [(r'\b(fixture|gameweek|gw|when.*play|schedule|against)\b', 2.5)],
            'position_analysis': [
                (r'\b(top|best|highest|list)\b.*\b(defenders?|midfielders?|forwards?|goalkeepers?)\b', 2.2),
                (r'\b(def|mid|fwd|gk)\b', 2.0),
                (r'\b(more than|less than|have|with)\b', 2.0)
            ],
            'recommendation': [(r'\b(recommend|suggest|should i|captain|pick)\b', 3.0)]
        }

        self.position_map = {
            'defender':'DEF', 'defenders':'DEF', 'defence':'DEF', 'def':'DEF',
            'midfielder':'MID', 'midfielders':'MID', 'midfield':'MID', 'mid':'MID',
            'forward':'FWD', 'forwards':'FWD', 'striker':'FWD', 'fwd':'FWD',
            'goalkeeper':'GK', 'goalkeepers':'GK', 'gk':'GK'
        }

        try: self.embedding_model = SentenceTransformer(embedding_model_name)
        except: self.embedding_model = None

    def classify_intent(self, query: str) -> Tuple[str, float]:
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in self.intent_patterns}
        for intent, patterns in self.intent_patterns.items():
            for pat, weight in patterns:
                if re.search(pat, query_lower): scores[intent] += weight
        if max(scores.values()) == 0: return 'general_query', 0.5
        return max(scores, key=scores.get), 0.8

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {'player_name': [], 'team_name': [], 'position': [], 'gameweek': [], 'numeric_value': [], 'season': []}
        query_lower = query.lower()

        if self.valid_players:
            for db_player in self.valid_players:
                db_p_lower = db_player.lower()
                if db_p_lower in query_lower:
                    entities['player_name'].append(db_player)
                    continue
                parts = db_p_lower.split()
                if len(parts) > 1 and re.search(r'\b' + re.escape(parts[-1]) + r'\b', query_lower):
                    entities['player_name'].append(db_player)

        if self.valid_teams:
            for db_team in self.valid_teams:
                if db_team.lower() in query_lower: entities['team_name'].append(db_team)

        for word, code in self.position_map.items():
            if re.search(r'\b' + re.escape(word) + r'\b', query_lower): entities['position'].append(code)
        
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            entities['numeric_value'] = numbers
            if 'gw' in query_lower or 'gameweek' in query_lower: entities['gameweek'] = numbers

        seasons = re.findall(r'\b(202[0-9]-2[0-9])\b', query)
        if seasons: entities['season'] = seasons

        for k in entities: entities[k] = list(set(entities[k]))
        return entities

    def generate_embedding(self, query: str) -> np.ndarray:
        if self.embedding_model: return self.embedding_model.encode(query, convert_to_numpy=True)
        return np.zeros(384)

    def build_cypher_params(self, entities: Dict[str, List[str]], intent: str) -> Dict[str, any]:
        p = {}
        if entities['player_name']:
            sorted_p = sorted(entities['player_name'], key=len, reverse=True)
            p['player_name'] = sorted_p[0]
            p['player_names'] = sorted_p
            if len(sorted_p) >= 2:
                p['player1'] = sorted_p[0]; p['player2'] = sorted_p[1]
        
        if entities['team_name']: p['team_name'] = entities['team_name'][0]
        if entities['position']: p['position'] = entities['position'][0]
        if entities['season']: p['season'] = entities['season'][0]
        if entities['gameweek']: 
            try: p['gameweek'] = int(entities['gameweek'][0])
            except: pass
        p['threshold'] = 0
        if entities['numeric_value']:
            try: p['threshold'] = int(entities['numeric_value'][0])
            except: pass
        return p

    def process(self, query: str) -> ProcessedInput:
        intent, conf = self.classify_intent(query)
        entities = self.extract_entities(query)
        params = self.build_cypher_params(entities, intent)
        return ProcessedInput(query, intent, conf, entities, None, params)

PREPROC = FPLInputPreprocessor(KG_TEAMS, KG_PLAYERS)

# ------------------------------------------------------
# 3.  Cypher Templates
# ------------------------------------------------------
CYPHER_TEMPLATES = {
    'player_performance': f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE toLower(p.{PLAYER_PROP}) CONTAINS toLower($player_name)
        RETURN  p.{PLAYER_PROP} AS player, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists, SUM(r.minutes) AS minutes, COUNT(f) AS matches
    """,
    'player_comparison': f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE p.{PLAYER_PROP} IN $player_names
        RETURN  p.{PLAYER_PROP} AS player, SUM(r.total_points) AS points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
        ORDER BY points DESC
    """,
    'team_analysis': f"""
        MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture)
        WHERE toLower(t.{TEAM_PROP}) CONTAINS toLower($team_name)
        WITH f
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        RETURN  COUNT(DISTINCT f) AS matches, SUM(r.goals_scored) AS goals_scored, SUM(r.clean_sheets) AS clean_sheets
    """,
    'fixture_query': f"""
        MATCH (f:Fixture)-[:HAS_HOME_TEAM]->(t1:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(t2:Team)
        WHERE f.gameweek = $gameweek
        RETURN  t1.{TEAM_PROP} AS home, t2.{TEAM_PROP} AS away, f.gameweek AS gameweek
        LIMIT 10
    """,
    'position_analysis': f"""
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {{name: $position}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, SUM(r.total_points) AS tp, SUM(r.goals_scored) AS g
        WHERE g >= $threshold
        ORDER BY tp DESC LIMIT 10
        RETURN  p.{PLAYER_PROP} AS player, tp AS total_points, g AS goals
    """,
    'general_query': f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, SUM(r.total_points) AS tp
        ORDER BY tp DESC LIMIT 5
        RETURN  p.{PLAYER_PROP} AS player, tp AS total_points
    """
}

# ------------------------------------------------------
# 4.  Readable Response Generator (The Logic You Need)
# ------------------------------------------------------
def format_response(intent, data, model_name):
    if not data or "error" in str(data):
        return "I couldn't find any relevant data in the Knowledge Graph for your query."

    # Generate natural language based on intent
    response = ""
    
    if intent == 'player_performance':
        # Single player stats
        row = data[0]
        response = f"**{row.get('player')}** has played {row.get('matches')} matches, scoring **{row.get('goals')} goals** and providing **{row.get('assists')} assists**, totaling {row.get('total_points')} points."

    elif intent == 'player_comparison':
        # Comparison table/list
        response = "**Player Comparison:**\n"
        for row in data:
            response += f"- **{row.get('player')}**: {row.get('points')} pts | {row.get('goals')} goals | {row.get('assists')} assists\n"

    elif intent == 'team_analysis':
        row = data[0]
        response = f"The team has scored **{row.get('goals_scored')} goals** and kept **{row.get('clean_sheets')} clean sheets** in {row.get('matches')} matches."

    elif intent == 'fixture_query':
        response = f"**Fixtures for Gameweek {data[0].get('gameweek')}:**\n"
        for row in data:
            response += f"- {row.get('home')} vs {row.get('away')}\n"

    elif intent in ['position_analysis', 'recommendation', 'general_query']:
        # Top lists
        response = "**Top Players Found:**\n"
        for i, row in enumerate(data, 1):
            response += f"{i}. **{row.get('player')}** - {row.get('total_points')} pts ({row.get('goals', 0)} goals)\n"
    
    else:
        # Fallback
        response = "Here is the data found:\n" + str(data)

    return f"{response}\n\n*(Generated by {model_name} simulation)*"

def build_cypher(intent, entities):       
    params = PREPROC.build_cypher_params(entities, intent)
    tmpl   = CYPHER_TEMPLATES.get(intent, CYPHER_TEMPLATES['general_query'])
    return tmpl, params

# ============================================================================
# STREAMLIT APP
# ============================================================================
st.set_page_config(page_title="Graph-RAG FPL Assistant", layout="wide")
if "history" not in st.session_state: st.session_state.history = []

with st.sidebar:
    st.title("⚙️ FPL Assistant")
    model = st.selectbox("LLM", ["gpt-3.5", "gpt-4", "llama-3"])
    show_cypher = st.checkbox("Debug Mode", value=True)
    if st.button("Clear History"): st.session_state.history = []
    st.divider()
    st.success(f"Schema: Player=`{PLAYER_PROP}`, Team=`{TEAM_PROP}`")
    st.info(f"Loaded: {len(KG_PLAYERS)} Players")

st.title("⚽ Graph-RAG FPL Assistant")
query = st.text_input("Ask about FPL:", placeholder="e.g. How many goals did Salah score?")

if st.button("Run") and query:
    with st.spinner("Processing..."):
        processed = PREPROC.process(query)
        cypher, params = build_cypher(processed.intent, processed.entities)
        
        kg_res = NEO.run(cypher, params)
        
        # USE THE NEW FORMATTER
        readable_answer = format_response(processed.intent, kg_res, model)
        
        st.session_state.history.append({
            "q": query, 
            "intent": processed.intent, 
            "cypher": cypher, 
            "params": params, 
            "kg": kg_res, 
            "answer": readable_answer
        })

for turn in reversed(st.session_state.history):
    st.markdown(f"### 🧑 {turn['q']}")
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown("**Knowledge Graph Data**")
        st.json(turn["kg"], expanded=False)
    with c2:
        st.markdown("**Answer**")
        st.info(turn["answer"])
    
    if show_cypher:
        with st.expander("Technical Details"):
            st.code(turn["cypher"], language="cypher")
            st.write(f"Intent: `{turn['intent']}`")
    st.divider()