# fpl_preproc.py – reusable preprocessing module
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

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
        # intent keywords
        self.intent_patterns = {
            'player_performance': [r'\b(goals?|assists?|points?|stats?|performance|how many|how much)\b'],
            'player_comparison':  [r'\b(compare|versus|vs\.?|better|who.*better)\b'],
            'team_analysis':      [r'\b(team|defence|attack|clean sheets)\b'],
            'fixture_query':      [r'\b(fixture|gameweek|gw|when.*play)\b'],
            'position_analysis':  [r'\b(top|best).*?\b(defenders?|midfielders?|forwards?|goalkeepers?|def|mid|fwd|gk)\b'],
            'recommendation':     [r'\b(recommend|suggest|should i|captain)\b'],
            'search_player':      [r'\b(tell me about|who is|info about)\b'],
        }
        # entity regex
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

    # ---- intent ----
    def classify_intent(self, query: str) -> Tuple[str, float]:
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in self.intent_patterns}
        for intent, patterns in self.intent_patterns.items():
            for pat in patterns:
                scores[intent] += len(re.findall(pat, query_lower))
        if max(scores.values()) == 0: return 'general_query', 0.5
        best = max(scores, key=scores.get)
        return best, min(1.0, scores[best] / len(self.intent_patterns[best]))

    # ---- entities ----
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

    # ---- embedding ----
    def generate_embedding(self, query: str) -> np.ndarray:
        return self.embedding_model.encode(query, convert_to_numpy=True)

    # ---- cypher params ----
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

    # ---- full pipeline ----
    def process(self, query: str) -> ProcessedInput:
        intent, conf = self.classify_intent(query)
        entities = self.extract_entities(query)
        emb = self.generate_embedding(query)
        params = self.build_cypher_params(entities, intent)
        return ProcessedInput(query, intent, conf, entities, emb, params)