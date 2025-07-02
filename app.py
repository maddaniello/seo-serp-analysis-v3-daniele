import streamlit as st
import requests
import pandas as pd
import time
import json
from collections import Counter, defaultdict
from urllib.parse import urlparse
from openai import OpenAI
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import aiohttp
import concurrent.futures
import threading
from functools import lru_cache
import re

# Configurazione della pagina
st.set_page_config(
    page_title="SERP Analyzer Pro - ScrapingDog",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .ai-overview-box {
        background-color: #e8f4fd;
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SERPAnalyzer:
    def __init__(self, scrapingdog_api_key, openai_api_key):
        self.scrapingdog_api_key = scrapingdog_api_key
        self.openai_api_key = openai_api_key
        self.scrapingdog_url = "https://api.scrapingdog.com/google"
        self.scrapingdog_ai_url = "https://api.scrapingdog.com/google/ai_overview"
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key != "dummy" else None
        self.classification_cache = {}
        self.use_ai = True
        self.batch_size = 5

    def fetch_serp_results(self, query, country="us", language="en", num_results=10, use_advance_search=True):
        """Effettua la ricerca SERP tramite ScrapingDog API"""
        params = {
            "api_key": self.scrapingdog_api_key,
            "query": query,
            "results": num_results,
            "country": country,
            "language": language,
            "page": 0
        }
        
        # Aggiungi advance_search solo se richiesto
        if use_advance_search:
            params["advance_search"] = "true"
        
        try:
            response = requests.get(self.scrapingdog_url, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error(f"⚠️ Errore 401 - ScrapingDog API key non valida per query '{query}'")
                return None
            elif response.status_code == 403:
                st.error(f"⚠️ Errore 403 - Crediti ScrapingDog esauriti per query '{query}'")
                return None
            elif response.status_code == 429:
                st.error(f"⚠️ Errore 429 - Troppe richieste, rallenta per query '{query}'")
                return None
            else:
                st.error(f"Errore ScrapingDog per query '{query}': {response.status_code}")
                # Mostra anche il contenuto della risposta per debug
                try:
                    error_content = response.json()
                    st.error(f"Dettaglio errore: {error_content}")
                except:
                    st.error(f"Contenuto errore: {response.text[:200]}")
                return None
        except Exception as e:
            st.error(f"Errore di connessione ScrapingDog: {e}")
            return None

    def fetch_ai_overview_details(self, ai_overview_url):
        """Fetch dettagli AI Overview usando l'URL dedicato di ScrapingDog"""
        if not ai_overview_url:
            return None
            
        params = {
            "api_key": self.scrapingdog_api_key,
            "url": ai_overview_url
        }
        
        try:
            response = requests.get(self.scrapingdog_ai_url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            return None

    @lru_cache(maxsize=1000)
    def classify_page_type_rule_based(self, url, title, snippet=""):
        """Classificazione veloce basata su regole per casi comuni"""
        url_lower = url.lower()
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        # Homepage patterns
        if (url_lower.count('/') <= 3 and 
            ('home' in url_lower or url_lower.endswith('.com') or url_lower.endswith('.it') or
             'homepage' in title_lower or 'home page' in title_lower)):
            return "Homepage"
        
        # Product page patterns
        product_patterns = ['product', 'prodotto', 'item', 'articolo', '/p/', 'buy', 'acquista', 'shop']
        if any(pattern in url_lower for pattern in product_patterns):
            return "Pagina Prodotto"
        
        # Category page patterns  
        category_patterns = ['category', 'categoria', 'catalogo', 'catalog', 'collection', 'collezione', 'products', 'prodotti']
        if any(pattern in url_lower for pattern in category_patterns):
            return "Pagina di Categoria"
            
        # Blog patterns
        blog_patterns = ['blog', 'news', 'notizie', 'articolo', 'post', 'article', '/blog/', 'magazine']
        if any(pattern in url_lower for pattern in blog_patterns):
            return "Articolo di Blog"
            
        # Services patterns
        service_patterns = ['service', 'servizio', 'servizi', 'services', 'consulenza', 'consulting']
        if any(pattern in url_lower for pattern in service_patterns):
            return "Pagina di Servizi"
            
        return None

    def classify_page_type_gpt(self, url, title, snippet=""):
        """Classificazione con OpenAI solo per casi complessi"""
        # Prima prova la classificazione rule-based
        rule_based_result = self.classify_page_type_rule_based(url, title, snippet)
        if rule_based_result:
            return rule_based_result
            
        # Cache check
        cache_key = f"{url}_{title}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Prompt ottimizzato per velocità
        prompt = f"""Classifica SOLO con una di queste categorie:
        
URL: {url}
Titolo: {title}

Categorie: Homepage, Pagina di Categoria, Pagina Prodotto, Articolo di Blog, Pagina di Servizi, Altro

Rispondi solo con la categoria."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            self.classification_cache[cache_key] = result
            return result
        except Exception as e:
            st.warning(f"Errore OpenAI: {e}")
            return "Altro"

    def classify_batch_openai(self, pages_data):
        """Classificazione in batch per ridurre le chiamate API"""
        if not pages_data or not self.use_ai or not self.client:
            return {}
            
        # Raggruppa per classificazione batch
        batch_size = min(len(pages_data), self.batch_size)
        batch_prompt = "Classifica ogni pagina con una di queste categorie: Homepage, Pagina di Categoria, Pagina Prodotto, Articolo di Blog, Pagina di Servizi, Altro\n\n"
        
        for i, (url, title, snippet) in enumerate(pages_data[:batch_size]):
            batch_prompt += f"{i+1}. URL: {url}\n   Titolo: {title}\n\n"
        
        batch_prompt += "Rispondi nel formato: 1. Categoria, 2. Categoria, ecc."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": batch_prompt}],
                max_tokens=100,
                temperature=0
            )
            
            # Parse della risposta batch
            results = {}
            response_text = response.choices[0].message.content.strip()
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                if str(i+1) in line and i < len(pages_data):
                    for category in ["Homepage", "Pagina di Categoria", "Pagina Prodotto", 
                                   "Articolo di Blog", "Pagina di Servizi", "Altro"]:
                        if category in line:
                            url, title, snippet = pages_data[i]
                            cache_key = f"{url}_{title}"
                            results[cache_key] = category
                            break
            
            return results
        except Exception as e:
            st.warning(f"Errore batch OpenAI: {e}")
            return {}

    def parse_ai_overview(self, data):
        """Estrae informazioni dall'AI Overview secondo ScrapingDog API"""
        ai_overview_info = {
            "has_ai_overview": False,
            "ai_overview_text": "",
            "ai_sources": [],
            "ai_source_domains": [],
            "ai_overview_url": None
        }
        
        # Controlla se c'è AI Overview embedded nei risultati
        if "ai_overview" in data:
            ai_data = data["ai_overview"]
            ai_overview_info["has_ai_overview"] = True
            
            # Estrai testo dai text_blocks
            if "text_blocks" in ai_data:
                text_parts = []
                for block in ai_data["text_blocks"]:
                    if "snippet" in block:
                        text_parts.append(block["snippet"])
                    
                    # Se è una lista, estrai anche gli elementi
                    if block.get("type") == "list" and "list" in block:
                        for item in block["list"]:
                            if "snippet" in item:
                                text_parts.append(f"• {item['snippet']}")
                
                ai_overview_info["ai_overview_text"] = " ".join(text_parts)
            
            # Estrai references (fonti)
            if "references" in ai_data:
                for ref in ai_data["references"]:
                    source_info = {
                        "title": ref.get("title", ""),
                        "link": ref.get("link", ""),
                        "domain": urlparse(ref.get("link", "")).netloc if ref.get("link") else "",
                        "source": ref.get("source", ""),
                        "snippet": ref.get("snippet", "")
                    }
                    ai_overview_info["ai_sources"].append(source_info)
                    if source_info["domain"]:
                        ai_overview_info["ai_source_domains"].append(source_info["domain"])
        
        # Cerca URL per AI Overview separato (ScrapingDog potrebbe fornirlo diversamente)
        # Controlla se c'è un URL dedicato per AI Overview
        ai_url_fields = ["ai_overview_url", "ai_url", "overview_url"]
        for field in ai_url_fields:
            if field in data:
                ai_overview_info["ai_overview_url"] = data[field]
                break
        
        # Se c'è un URL ma non abbiamo ancora l'AI Overview, prova la chiamata separata
        if ai_overview_info["ai_overview_url"] and not ai_overview_info["has_ai_overview"]:
            detailed_ai = self.fetch_ai_overview_details(ai_overview_info["ai_overview_url"])
            if detailed_ai and "ai_overview" in detailed_ai:
                return self.parse_ai_overview(detailed_ai)
        
        # Fallback: cerca in answer_box come AI Overview alternativo
        if not ai_overview_info["has_ai_overview"] and "answer_box" in data:
            answer_box = data["answer_box"]
            if isinstance(answer_box, dict):
                ai_overview_info["has_ai_overview"] = True
                ai_overview_info["ai_overview_text"] = str(answer_box.get("snippet", answer_box.get("answer", "")))
                
                if "link" in answer_box:
                    source_info = {
                        "title": answer_box.get("title", ""),
                        "link": answer_box.get("link", ""),
                        "domain": urlparse(answer_box.get("link", "")).netloc if answer_box.get("link") else ""
                    }
                    ai_overview_info["ai_sources"].append(source_info)
                    if source_info["domain"]:
                        ai_overview_info["ai_source_domains"].append(source_info["domain"])
        
        return ai_overview_info

    def debug_response_structure(self, data, query):
        """Debug della struttura della risposta ScrapingDog per capire dove sono i dati AI Overview"""
        st.write(f"🔍 **Debug struttura dati ScrapingDog per query: {query}**")
        st.write("**Chiavi principali trovate:**")
        for key in data.keys():
            st.write(f"- {key}: {type(data[key])}")
        
        # Controlla specificamente AI Overview
        if "ai_overview" in data:
            st.write("**🤖 AI Overview trovato!**")
            ai_data = data["ai_overview"]
            st.write(f"- Tipo: {type(ai_data)}")
            if isinstance(ai_data, dict):
                st.write("- Sottocampi:")
                for subkey in ai_data.keys():
                    st.write(f"  - {subkey}: {type(ai_data[subkey])}")
                
                # Mostra text_blocks se presenti
                if "text_blocks" in ai_data:
                    st.write(f"  - text_blocks contiene {len(ai_data['text_blocks'])} blocchi")
                
                # Mostra references se presenti
                if "references" in ai_data:
                    st.write(f"  - references contiene {len(ai_data['references'])} fonti")
                    for i, ref in enumerate(ai_data['references'][:3]):  # Prime 3
                        st.write(f"    {i+1}. {ref.get('title', 'No title')} - {ref.get('source', 'No source')}")
        
        # Controlla organic_data (ScrapingDog)
        if "organic_data" in data:
            st.write(f"**📊 organic_data trovato: {len(data['organic_data'])} risultati**")
        
        # Controlla people_also_ask (ScrapingDog)
        if "people_also_ask" in data:
            st.write(f"**❓ people_also_ask trovato: {len(data['people_also_ask'])} domande**")
            for i, paa in enumerate(data['people_also_ask'][:3]):  # Prime 3
                st.write(f"    {i+1}. {paa.get('question', 'No question')}")
        
        # Controlla altri campi che potrebbero contenere AI Overview
        other_ai_fields = ["answer_box", "featured_snippet", "knowledge_graph"]
        for field in other_ai_fields:
            if field in data:
                st.write(f"**📦 {field} trovato:**")
                field_data = data[field]
                if isinstance(field_data, dict):
                    for subkey in field_data.keys():
                        st.write(f"  - {subkey}: {type(field_data[subkey])}")
        
        # Mostra alcuni campioni di dati se richiesto
        if st.checkbox(f"Mostra dati JSON completi per '{query}'", key=f"debug_full_{query}"):
            st.json(data)

    def cluster_keywords_with_custom(self, keywords, custom_clusters):
        """Clusterizza le keyword usando cluster personalizzati come priorità"""
        if not self.client or not self.use_ai:
            return self.cluster_keywords_simple_custom(keywords, custom_clusters)
        
        # Dividi in batch per evitare prompt troppo lunghi
        batch_size = 50
        all_clusters = {}
        
        # Inizializza cluster personalizzati
        for cluster_name in custom_clusters:
            all_clusters[cluster_name] = []
        
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i+batch_size]
            
            prompt = f"""Ruolo: Esperto di analisi semantica e architettura siti web
Capacità: Specialista in clustering di keyword basato su strutture di siti web esistenti.

Compito: Assegna ogni keyword al cluster più appropriato, dando PRIORITÀ ai cluster predefiniti del sito.

CLUSTER PREDEFINITI (USA QUESTI COME PRIORITÀ):
{chr(10).join([f"- {cluster}" for cluster in custom_clusters])}

Keyword da classificare:
{chr(10).join([f"- {kw}" for kw in batch_keywords])}

Istruzioni:
1. PRIORITÀ ASSOLUTA: Cerca di assegnare ogni keyword a uno dei cluster predefiniti se semanticamente correlata
2. Solo se una keyword NON può essere associata a nessun cluster predefinito, crea un nuovo cluster
3. Ogni cluster deve avere almeno 3 keyword (per quelli nuovi)
4. Se una keyword non si adatta a nessun cluster, mettila in "Generale"

Formato di risposta:
Cluster: [Nome Cluster Predefinito o Nuovo]
- keyword1
- keyword2
- keyword3

Cluster: [Altro Cluster]
- keyword4
- keyword5"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                
                # Parse della risposta
                response_text = response.choices[0].message.content.strip()
                clusters = self.parse_clustering_response_custom(response_text, custom_clusters)
                
                # Merge dei risultati
                for cluster_name, cluster_keywords in clusters.items():
                    if cluster_name in all_clusters:
                        all_clusters[cluster_name].extend(cluster_keywords)
                    else:
                        all_clusters[cluster_name] = cluster_keywords
                
            except Exception as e:
                st.warning(f"Errore clustering personalizzato batch {i//batch_size + 1}: {e}")
                # Fallback per questo batch
                simple_clusters = self.cluster_keywords_simple_custom(batch_keywords, custom_clusters)
                for cluster_name, cluster_keywords in simple_clusters.items():
                    if cluster_name in all_clusters:
                        all_clusters[cluster_name].extend(cluster_keywords)
                    else:
                        all_clusters[cluster_name] = cluster_keywords
        
        # Pulisci cluster vuoti
        final_clusters = {k: v for k, v in all_clusters.items() if v}
        
        return final_clusters

    def cluster_keywords_simple_custom(self, keywords, custom_clusters):
        """Clustering semplice con cluster personalizzati (fallback)"""
        clusters = {}
        
        # Inizializza cluster personalizzati
        for cluster_name in custom_clusters:
            clusters[cluster_name] = []
        
        unassigned_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            assigned = False
            
            # Prova ad assegnare a cluster personalizzati
            for cluster_name in custom_clusters:
                cluster_words = cluster_name.lower().split()
                if any(word in keyword_lower or keyword_lower in word for word in cluster_words):
                    clusters[cluster_name].append(keyword)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_keywords.append(keyword)
        
        # Raggruppa keyword non assegnate
        if unassigned_keywords:
            auto_clusters = self.cluster_keywords_simple(unassigned_keywords)
            clusters.update(auto_clusters)
        
        # Rimuovi cluster vuoti
        final_clusters = {k: v for k, v in clusters.items() if v}
        
        return final_clusters

    def parse_clustering_response_custom(self, response_text, custom_clusters):
        """Parse della risposta di clustering personalizzato"""
        clusters = {}
        current_cluster = None
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Cluster:'):
                current_cluster = line.replace('Cluster:', '').strip()
                clusters[current_cluster] = []
            elif line.startswith('-') and current_cluster:
                keyword = line.replace('-', '').strip()
                if keyword:
                    clusters[current_cluster].append(keyword)
        
        # Per cluster personalizzati, accetta anche cluster con meno di 5 keyword
        # ma per quelli nuovi mantieni il minimo
        valid_clusters = {}
        small_keywords = []
        
        for cluster_name, keywords in clusters.items():
            if cluster_name in custom_clusters:
                # Cluster personalizzati: accetta qualsiasi size
                valid_clusters[cluster_name] = keywords
            elif len(keywords) >= 3:
                # Cluster nuovi: minimo 3 keyword
                valid_clusters[cluster_name] = keywords
            else:
                small_keywords.extend(keywords)
        
        if small_keywords:
            if "Generale" in valid_clusters:
                valid_clusters["Generale"].extend(small_keywords)
            else:
                valid_clusters["Generale"] = small_keywords
        
        return valid_clusters

    def cluster_keywords_semantic(self, keywords):
        """Clusterizza le keyword per gruppi semantici usando OpenAI"""
        if not self.client or not self.use_ai:
            return self.cluster_keywords_simple(keywords)
        
        batch_size = 50
        all_clusters = {}
        
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i+batch_size]
            
            prompt = f"""Ruolo: Esperto di analisi semantica
Capacità: Possiedi competenze approfondite in linguistica computazionale, analisi semantica e clustering di parole chiave.

Compito: Clusterizza il seguente elenco di keyword raggruppando quelle appartenenti allo stesso gruppo semantico. Ogni cluster deve contenere ALMENO 5 keyword per essere valido.

Elenco keyword da analizzare:
{chr(10).join([f"- {kw}" for kw in batch_keywords])}

Istruzioni:
1. Raggruppa le keyword per similarità semantica, significato e contesto d'uso
2. Ogni cluster deve avere almeno 5 keyword
3. Se una keyword non ha abbastanza correlate, inseriscila nel cluster "Generale"
4. Dai un nome descrittivo a ogni cluster

Formato di risposta:
Cluster: [Nome Cluster]
- keyword1
- keyword2
- keyword3
- keyword4
- keyword5

Cluster: [Nome Cluster 2]
- keyword6
- keyword7
[etc...]"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                response_text = response.choices[0].message.content.strip()
                clusters = self.parse_clustering_response(response_text)
                all_clusters.update(clusters)
                
            except Exception as e:
                st.warning(f"Errore clustering OpenAI batch {i//batch_size + 1}: {e}")
                simple_clusters = self.cluster_keywords_simple(batch_keywords)
                all_clusters.update(simple_clusters)
        
        return all_clusters

    def cluster_keywords_simple(self, keywords):
        """Clustering semplice basato su parole comuni (fallback)"""
        clusters = defaultdict(list)
        
        for keyword in keywords:
            words = keyword.lower().split()
            main_word = words[0] if words else keyword
            
            assigned = False
            for cluster_name in clusters:
                if any(word in cluster_name.lower() or cluster_name.lower() in word for word in words):
                    clusters[cluster_name].append(keyword)
                    assigned = True
                    break
            
            if not assigned:
                clusters[f"Cluster {main_word.capitalize()}"].append(keyword)
        
        final_clusters = {}
        small_clusters = []
        
        for cluster_name, cluster_keywords in clusters.items():
            if len(cluster_keywords) >= 5:
                final_clusters[cluster_name] = cluster_keywords
            else:
                small_clusters.extend(cluster_keywords)
        
        if small_clusters:
            final_clusters["Generale"] = small_clusters
        
        return final_clusters

    def parse_clustering_response(self, response_text):
        """Parse della risposta di clustering da OpenAI"""
        clusters = {}
        current_cluster = None
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Cluster:'):
                current_cluster = line.replace('Cluster:', '').strip()
                clusters[current_cluster] = []
            elif line.startswith('-') and current_cluster:
                keyword = line.replace('-', '').strip()
                if keyword:
                    clusters[current_cluster].append(keyword)
        
        valid_clusters = {}
        small_keywords = []
        
        for cluster_name, keywords in clusters.items():
            if len(keywords) >= 5:
                valid_clusters[cluster_name] = keywords
            else:
                small_keywords.extend(keywords)
        
        if small_keywords:
            valid_clusters["Generale"] = small_keywords
        
        return valid_clusters

    def parse_results(self, data, query):
        """Analizza i risultati SERP con classificazione ottimizzata e AI Overview per ScrapingDog"""
        domain_page_types = defaultdict(lambda: defaultdict(int))
        domain_occurences = defaultdict(int)
        query_page_types = defaultdict(list)
        paa_questions = []
        related_queries = []
        paa_to_queries = defaultdict(set)
        related_to_queries = defaultdict(set)
        paa_to_domains = defaultdict(set)

        pages_to_classify = []
        pages_info = []
        
        # Analizza AI Overview
        ai_overview_info = self.parse_ai_overview(data)
        
        # Analizza risultati organici (ScrapingDog usa "organic_data")
        if "organic_data" in data:
            for result in data["organic_data"]:
                domain = urlparse(result["link"]).netloc
                url = result["link"]
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                page_type = self.classify_page_type_rule_based(url, title, snippet)
                
                if page_type:
                    domain_page_types[domain][page_type] += 1
                    domain_occurences[domain] += 1
                    query_page_types[query].append(page_type)
                else:
                    pages_to_classify.append((url, title, snippet))
                    pages_info.append((domain, url, title, snippet))

        # Classificazione AI per pagine non classificate con regole
        if pages_to_classify and self.use_ai:
            batch_results = self.classify_batch_openai(pages_to_classify)
            
            for domain, url, title, snippet in pages_info:
                cache_key = f"{url}_{title}"
                page_type = batch_results.get(cache_key, "Altro")
                
                domain_page_types[domain][page_type] += 1
                domain_occurences[domain] += 1
                query_page_types[query].append(page_type)
        elif pages_to_classify and not self.use_ai:
            for domain, url, title, snippet in pages_info:
                page_type = "Altro"
                domain_page_types[domain][page_type] += 1
                domain_occurences[domain] += 1
                query_page_types[query].append(page_type)

        # Analizza People Also Ask (ScrapingDog usa "people_also_ask")
        if "people_also_ask" in data:
            for paa in data["people_also_ask"]:
                paa_text = paa.get("question", "")
                if paa_text:
                    paa_questions.append(paa_text)
                    paa_to_queries[paa_text].add(query)
                    paa_to_domains[paa_text].update([domain for domain in domain_page_types.keys()])

        # Analizza Related Searches (se presente in ScrapingDog)
        if "related_searches" in data:
            for related in data["related_searches"]:
                if isinstance(related, dict):
                    related_text = related.get("query", "")
                elif isinstance(related, str):
                    related_text = related
                else:
                    continue
                    
                if related_text:
                    related_queries.append(related_text)
                    related_to_queries[related_text].add(query)

        return (domain_page_types, domain_occurences, query_page_types, 
                paa_questions, related_queries, paa_to_queries, 
                related_to_queries, paa_to_domains, ai_overview_info)

    def create_excel_report(self, domains_counter, domain_occurences, query_page_types, 
                           domain_page_types, paa_questions, related_queries, 
                           paa_to_queries, related_to_queries, paa_to_domains, 
                           ai_overview_data, keyword_clusters=None):
        """Crea il report Excel con informazioni su AI Overview"""
        
        domain_page_types_list = []
        page_type_counter = Counter()

        for domain, page_type_dict in domain_page_types.items():
            domain_data = {
                "Competitor": domain, 
                "Numero occorrenze": domain_occurences[domain]
            }
            
            for page_type in ['Homepage', 'Pagina di Categoria', 'Pagina Prodotto', 
                            'Articolo di Blog', 'Pagina di Servizi', 'Altro']:
                domain_data[page_type] = page_type_dict.get(page_type, 0)
                page_type_counter[page_type] += domain_data[page_type]
            
            domain_page_types_list.append(domain_data)

        domain_page_types_df = pd.DataFrame(domain_page_types_list)

        domains_df = pd.DataFrame(domains_counter.items(), columns=["Dominio", "Occorrenze"])
        
        # Gestisci il caso quando non ci sono domini (evita TypeError)
        if not domains_df.empty and len(domains_counter) > 0:
            total_queries = sum(domains_counter.values())
            if total_queries > 0:
                domains_df["% Presenza"] = (domains_df["Occorrenze"] / total_queries * 100).round(2)
            else:
                domains_df["% Presenza"] = 0.0
        else:
            # Crea DataFrame vuoto con colonne corrette
            domains_df = pd.DataFrame(columns=["Dominio", "Occorrenze", "% Presenza"])

        query_page_type_data = []
        if query_page_types:
            for query, page_types in query_page_types.items():
                if page_types:  # Solo se ci sono page_types validi
                    for page_type, count in Counter(page_types).items():
                        query_page_type_data.append({
                            "Query": query, 
                            "Tipologia Pagina": page_type, 
                            "Occorrenze": count
                        })
        query_page_type_df = pd.DataFrame(query_page_type_data)

        # Gestisci PAA questions
        paa_data = []
        if paa_questions:
            unique_paa = list(set(paa_questions))  # Rimuovi duplicati
            for paa in unique_paa:
                paa_data.append({
                    "People Also Ask": paa,
                    "Keyword che lo attivano": ", ".join(paa_to_queries.get(paa, []))
                })
        paa_df = pd.DataFrame(paa_data)

        # Gestisci Related queries
        related_data = []
        if related_queries:
            unique_related = list(set(related_queries))  # Rimuovi duplicati
            for related in unique_related:
                related_data.append({
                    "Related Query": related,
                    "Keyword che lo attivano": ", ".join(related_to_queries.get(related, []))
                })
        related_df = pd.DataFrame(related_data)

        page_type_df = pd.DataFrame(page_type_counter.items(), 
                                  columns=["Tipologia Pagina", "Occorrenze"])

        # Crea DataFrame per AI Overview
        ai_overview_list = []
        ai_sources_list = []
        
        if ai_overview_data:
            for query, ai_info in ai_overview_data.items():
                if ai_info:  # Verifica che ai_info non sia None o vuoto
                    ai_overview_list.append({
                        "Query": query,
                        "Ha AI Overview": ai_info.get("has_ai_overview", False),
                        "Testo AI Overview": (ai_info.get("ai_overview_text", "")[:500] + "...") if len(ai_info.get("ai_overview_text", "")) > 500 else ai_info.get("ai_overview_text", ""),
                        "Numero Fonti": len(ai_info.get("ai_sources", []))
                    })
                    
                    # Aggiungi fonti separate
                    for i, source in enumerate(ai_info.get("ai_sources", [])):
                        ai_sources_list.append({
                            "Query": query,
                            "Fonte #": i + 1,
                            "Titolo Fonte": source.get("title", ""),
                            "Link Fonte": source.get("link", ""),
                            "Dominio Fonte": source.get("domain", "")
                        })
        
        ai_overview_df = pd.DataFrame(ai_overview_list)
        ai_sources_df = pd.DataFrame(ai_sources_list)

        # Clustering DataFrame
        clustering_df = pd.DataFrame()
        if keyword_clusters:
            clustering_data = []
            for cluster_name, keywords in keyword_clusters.items():
                for keyword in keywords:
                    clustering_data.append({
                        "Cluster": cluster_name,
                        "Keyword": keyword
                    })
            clustering_df = pd.DataFrame(clustering_data)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            domains_df.to_excel(writer, sheet_name="Top Domains", index=False)
            page_type_df.to_excel(writer, sheet_name="Tipologie di Pagine", index=False)
            domain_page_types_df.to_excel(writer, sheet_name="Competitor e Tipologie", index=False)
            query_page_type_df.to_excel(writer, sheet_name="Tipologie per Query", index=False)
            ai_overview_df.to_excel(writer, sheet_name="AI Overview", index=False)
            ai_sources_df.to_excel(writer, sheet_name="AI Overview Sources", index=False)
            paa_df.to_excel(writer, sheet_name="People Also Ask", index=False)
            related_df.to_excel(writer, sheet_name="Related Queries", index=False)
            if not clustering_df.empty:
                clustering_df.to_excel(writer, sheet_name="Keyword Clustering", index=False)

        return output.getvalue(), domains_df, page_type_df, domain_page_types_df, clustering_df, ai_overview_df, ai_sources_df

def main():
    st.markdown('<h1 class="main-header">🔍 SERP Analyzer Pro con ScrapingDog</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.header("⚙️ Configurazione")
    
    scrapingdog_api_key = st.sidebar.text_input(
        "ScrapingDog API Key", 
        type="password",
        help="Inserisci la tua API key di ScrapingDog.com"
    )
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Inserisci la tua API key di OpenAI"
    )

    st.sidebar.subheader("🔑 Validazione API")
    if scrapingdog_api_key:
        st.sidebar.success("✅ ScrapingDog API Key inserita")
    else:
        st.sidebar.warning("⚠️ Inserisci ScrapingDog API Key")
    
    if openai_api_key:
        st.sidebar.success("✅ OpenAI API Key inserita")
    else:
        st.sidebar.info("💡 OpenAI opzionale per classificazione AI")
    
    # Test API Button
    if st.sidebar.button("🧪 Testa API", help="Testa ScrapingDog API con una query semplice"):
        if not scrapingdog_api_key:
            st.sidebar.error("⚠️ Inserisci prima la ScrapingDog API Key!")
        else:
            with st.sidebar:
                with st.spinner("Testando API..."):
                    test_analyzer = SERPAnalyzer(scrapingdog_api_key, "dummy")
                    test_result = test_analyzer.fetch_serp_results("test", "us", "en", 5, False)
                    
                    if test_result:
                        st.sidebar.success("✅ API funziona correttamente!")
                        st.sidebar.write(f"Chiavi trovate: {list(test_result.keys())}")
                    else:
                        st.sidebar.error("❌ API non funziona - controlla la key")

    st.sidebar.subheader("🌍 Parametri di Ricerca")
    country = st.sidebar.selectbox(
        "Paese",
        ["us", "it", "uk", "de", "fr", "es"],
        index=0,
        help="Seleziona il paese per la ricerca"
    )
    
    language = st.sidebar.selectbox(
        "Lingua",
        ["en", "it", "de", "fr", "es"],
        index=0,
        help="Seleziona la lingua dei risultati"
    )
    
    num_results = st.sidebar.slider(
        "Numero di risultati per query",
        min_value=5,
        max_value=20,
        value=10,
        help="Numero di risultati da analizzare per ogni query"
    )
    
    st.sidebar.subheader("⚡ Opzioni Velocità")
    use_ai_classification = st.sidebar.checkbox(
        "Usa AI per classificazione avanzata",
        value=True,
        help="Disabilita per analisi ultra-veloce (solo regole)"
    )
    
    use_advance_search = st.sidebar.checkbox(
        "🔍 Advance Search",
        value=True,
        help="Include AI Overview e feature avanzate (10 crediti vs 5)"
    )
    
    enable_keyword_clustering = st.sidebar.checkbox(
        "Abilita clustering semantico keyword",
        value=True,
        help="Raggruppa le keyword per gruppi semantici"
    )
    
    batch_size = st.sidebar.slider(
        "Dimensione batch AI",
        min_value=1,
        max_value=10,
        value=5,
        help="Pagine da classificare insieme (più alto = più veloce)"
    ) if use_ai_classification else 1
    
    enable_debug = st.sidebar.checkbox(
        "🐛 Modalità Debug",
        value=False,
        help="Mostra struttura dati ScrapingDog per debug"
    )
    
    test_single_query = st.sidebar.checkbox(
        "🧪 Test Singola Query",
        value=False,
        help="Testa prima una singola query per verificare l'API"
    )

    st.header("📝 Inserisci le Query")
    
    # Avviso sui crediti
    st.info("💰 **Costi ScrapingDog**: 5 crediti per query normale, 10 crediti con advance_search (per AI Overview)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        queries_input = st.text_area(
            "Query da analizzare (una per riga)",
            height=200,
            placeholder="Inserisci le tue keyword qui...\nUna per ogni riga\n\nEsempio:\ncorso python\ncorso programmazione\nlearn python online"
        )
    
    with col2:
        st.markdown("### 💡 Suggerimenti")
        st.info("""
        • Una query per riga
        • Massimo 1000 query
        • Evita caratteri speciali
        • Usa query specifiche per il tuo settore
        """)

    if enable_keyword_clustering:
        st.header("🏗️ Cluster Personalizzati (Opzionale)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_clusters_input = st.text_area(
                "Nomi delle pagine/categorie del tuo sito (una per riga)",
                height=150,
                placeholder="Inserisci i nomi delle tue pagine principali...\nUna per ogni riga\n\nEsempio:\nServizi SEO\nCorsi Online\nConsulenza Marketing\nBlog Aziendale\nChi Siamo"
            )
        
        with col2:
            st.markdown("### 🎯 Cluster Strategici")
            st.info("""
            • Nomi delle tue pagine principali
            • Categorie del sito
            • Servizi offerti
            • Sezioni importanti
            • Lascia vuoto per clustering automatico
            """)

    if st.button("🚀 Avvia Analisi", type="primary", use_container_width=True):
        if use_ai_classification and (not scrapingdog_api_key or not openai_api_key):
            st.error("⚠️ Inserisci entrambe le API keys per l'analisi AI!")
            return
        elif not use_ai_classification and not scrapingdog_api_key:
            st.error("⚠️ Inserisci almeno la ScrapingDog API key!")
            return
        
        if not queries_input.strip():
            st.error("⚠️ Inserisci almeno una query!")
            return

        queries = [q.strip() for q in queries_input.strip().split('\n') if q.strip()]
        
        if len(queries) > 1000:
            st.error("⚠️ Massimo 1000 query per volta!")
            return
        
        # Calcola crediti stimati
        credits_per_query = 10 if use_advance_search else 5
        estimated_credits = len(queries) * credits_per_query
        st.info(f"💰 **Crediti stimati necessari**: {estimated_credits} ({credits_per_query} per query{' con advance_search' if use_advance_search else ''})")
        
        if estimated_credits > 1000:
            st.warning(f"⚠️ Analisi richiede molti crediti ({estimated_credits}). Considera di ridurre le query o disattivare advance_search.")

        custom_clusters = []
        if enable_keyword_clustering and 'custom_clusters_input' in locals() and custom_clusters_input.strip():
            custom_clusters = [c.strip() for c in custom_clusters_input.strip().split('\n') if c.strip()]

        if use_ai_classification:
            analyzer = SERPAnalyzer(scrapingdog_api_key, openai_api_key)
            st.info("🤖 Modalità AI attivata - Classificazione avanzata delle pagine + AI Overview")
        else:
            analyzer = SERPAnalyzer(scrapingdog_api_key, "dummy")
            st.info("⚡ Modalità Veloce attivata - Solo classificazione basata su regole")
        
        analyzer.use_ai = use_ai_classification
        analyzer.batch_size = batch_size
        
        # Test singola query se richiesto
        if test_single_query and queries:
            st.info(f"🧪 **Test Singola Query**: {queries[0]}")
            test_result = analyzer.fetch_serp_results(queries[0], country, language, num_results, use_advance_search)
            
            if test_result:
                st.success("✅ Test riuscito! Procedo con l'analisi completa.")
                with st.expander("👀 Visualizza dati di test"):
                    st.write("**Chiavi principali trovate:**")
                    for key in test_result.keys():
                        st.write(f"- {key}: {type(test_result[key])}")
                    
                    if "organic_data" in test_result:
                        st.write(f"- organic_data contiene: {len(test_result['organic_data'])} risultati")
                    
                    if "people_also_ask" in test_result:
                        st.write(f"- people_also_ask contiene: {len(test_result['people_also_ask'])} domande")
                        
                    if enable_debug:
                        st.json(test_result)
            else:
                st.error("❌ Test fallito! Controlla l'API key e riprova.")
                return
        
        keyword_clusters = {}
        if enable_keyword_clustering and (use_ai_classification or len(queries) > 0):
            status_text = st.empty()
            
            if custom_clusters:
                status_text.text(f"🏗️ Clustering con {len(custom_clusters)} cluster personalizzati...")
                try:
                    keyword_clusters = analyzer.cluster_keywords_with_custom(queries, custom_clusters)
                    st.success(f"✅ Cluster creati: {len(keyword_clusters)} (inclusi {len([k for k in keyword_clusters.keys() if k in custom_clusters])} personalizzati)")
                    
                    with st.expander("👀 Preview Clustering Personalizzato"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Cluster Personalizzati Utilizzati:**")
                            for cluster_name in custom_clusters:
                                if cluster_name in keyword_clusters and keyword_clusters[cluster_name]:
                                    st.write(f"✅ **{cluster_name}** ({len(keyword_clusters[cluster_name])} keyword)")
                                else:
                                    st.write(f"⚪ **{cluster_name}** (nessuna keyword assegnata)")
                        
                        with col2:
                            st.write("**Cluster Aggiuntivi Creati:**")
                            additional_clusters = [k for k in keyword_clusters.keys() if k not in custom_clusters]
                            for cluster_name in additional_clusters[:5]:
                                st.write(f"🆕 **{cluster_name}** ({len(keyword_clusters[cluster_name])} keyword)")
                            if len(additional_clusters) > 5:
                                st.write(f"... e altri {len(additional_clusters) - 5} cluster")
                
                except Exception as e:
                    st.warning(f"Errore durante il clustering personalizzato: {e}")
                    keyword_clusters = {}
            else:
                status_text.text("🧠 Clustering semantico automatico delle keyword...")
                try:
                    keyword_clusters = analyzer.cluster_keywords_semantic(queries)
                    st.success(f"✅ Identificati {len(keyword_clusters)} cluster semantici!")
                    
                    with st.expander("👀 Preview Clustering Automatico"):
                        for cluster_name, keywords in list(keyword_clusters.items())[:3]:
                            st.write(f"**{cluster_name}** ({len(keywords)} keyword)")
                            st.write(", ".join(keywords[:10]) + ("..." if len(keywords) > 10 else ""))
                
                except Exception as e:
                    st.warning(f"Errore durante il clustering: {e}")
                    keyword_clusters = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_domains = []
        query_page_types = defaultdict(list)
        domain_page_types = defaultdict(lambda: defaultdict(int))
        domain_occurences = defaultdict(int)
        paa_questions = []
        related_queries = []
        paa_to_queries = defaultdict(set)
        related_to_queries = defaultdict(set)
        paa_to_domains = defaultdict(set)
        ai_overview_data = {}
        successful_queries = 0
        failed_queries = 0

        for i, query in enumerate(queries):
            status_text.text(f"🔍 Analizzando: {query} ({i+1}/{len(queries)})")
            
            try:
                results = analyzer.fetch_serp_results(query, country, language, num_results, use_advance_search)
                
                if results:
                    successful_queries += 1
                    
                    # Debug mode: mostra struttura dati
                    if enable_debug and i < 3:  # Solo per le prime 3 query per non sovraccaricare
                        with st.expander(f"🐛 Debug dati per '{query}'"):
                            analyzer.debug_response_structure(results, query)
                            
                            # Mostra anche il parsing AI Overview in tempo reale
                            st.write("**🤖 Parsing AI Overview per questa query:**")
                            ai_test = analyzer.parse_ai_overview(results)
                            st.write(f"- Ha AI Overview: {ai_test['has_ai_overview']}")
                            if ai_test['ai_overview_text']:
                                st.write(f"- Testo trovato: {ai_test['ai_overview_text'][:200]}...")
                            st.write(f"- Fonti trovate: {len(ai_test['ai_sources'])}")
                            if ai_test['ai_sources']:
                                for j, source in enumerate(ai_test['ai_sources'][:3]):
                                    st.write(f"  {j+1}. {source['title']} - {source['domain']}")
                    
                    (domain_page_types_query, domain_occurences_query, query_page_types_query,
                     paa_questions_query, related_queries_query, paa_to_queries_query,
                     related_to_queries_query, paa_to_domains_query, ai_overview_info) = analyzer.parse_results(results, query)
                    
                    # Salva info AI Overview
                    ai_overview_data[query] = ai_overview_info
                    
                    for domain, page_types in domain_page_types_query.items():
                        for page_type, count in page_types.items():
                            domain_page_types[domain][page_type] += count
                    
                    for domain, count in domain_occurences_query.items():
                        domain_occurences[domain] += count
                    
                    for query_key, page_types in query_page_types_query.items():
                        query_page_types[query_key].extend(page_types)
                    
                    paa_questions.extend(paa_questions_query)
                    related_queries.extend(related_queries_query)
                    paa_to_queries.update(paa_to_queries_query)
                    related_to_queries.update(related_to_queries_query)
                    paa_to_domains.update(paa_to_domains_query)
                    all_domains.extend(domain_page_types_query.keys())
                else:
                    failed_queries += 1
                    st.warning(f"⚠️ Nessun risultato per query: {query}")
            
            except Exception as e:
                failed_queries += 1
                st.error(f"⚠️ Errore durante l'analisi della query '{query}': {str(e)}")
                continue  # Continua con la query successiva
            
            progress_bar.progress((i + 1) / len(queries))
            
            sleep_time = 0.5 if not use_ai_classification else 1.0
            time.sleep(sleep_time)

        # Mostra statistiche finali
        st.info(f"📊 **Statistiche Analisi**: {successful_queries} query riuscite, {failed_queries} fallite su {len(queries)} totali")

        status_text.text("✅ Analisi completata! Generazione report...")

        domains_counter = Counter(all_domains)
        
        # Verifica se abbiamo almeno alcuni risultati validi
        if not domains_counter and successful_queries == 0:
            st.error("❌ **Nessun risultato valido trovato per nessuna query.**")
            st.info("💡 **Suggerimenti:**")
            st.info("- Controlla che la ScrapingDog API key sia corretta")
            st.info("- Verifica di avere crediti sufficienti (hai bisogno di almeno 5-10 crediti per query)")
            st.info("- Prova con query più semplici (es: 'pizza', 'news')")
            st.info("- Usa il bottone '🧪 Testa API' per verificare la connessione")
            return
        elif successful_queries < len(queries):
            st.warning(f"⚠️ **Analisi parziale completata**: {successful_queries}/{len(queries)} query riuscite")
            st.info("💡 Procedo con la generazione del report sui dati disponibili")
        
        try:
            excel_data, domains_df, page_type_df, domain_page_types_df, clustering_df, ai_overview_df, ai_sources_df = analyzer.create_excel_report(
                domains_counter, domain_occurences, query_page_types, domain_page_types,
                paa_questions, related_queries, paa_to_queries, related_to_queries, paa_to_domains, 
                ai_overview_data, keyword_clusters
            )
        except Exception as e:
            st.error(f"⚠️ Errore durante la generazione del report: {str(e)}")
            st.info("💡 L'analisi è stata completata ma ci sono stati problemi nella generazione del report Excel.")
            return

        status_text.text("📊 Visualizzazione risultati...")

        st.markdown("---")
        st.header("📊 Risultati Analisi")

        # Metriche principali
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Query Riuscite", successful_queries)
        with col2:
            success_rate = (successful_queries / len(queries)) * 100 if len(queries) > 0 else 0
            st.metric("Tasso Successo", f"{success_rate:.1f}%")
        with col3:
            st.metric("Domini Trovati", len(domains_counter))
        with col4:
            st.metric("PAA Questions", len(set(paa_questions)))
        with col5:
            cluster_count = len(keyword_clusters) if keyword_clusters else 0
            st.metric("Cluster Semantici", cluster_count)
        with col6:
            ai_overview_count = 0
            if ai_overview_data:
                ai_overview_count = sum(1 for ai_info in ai_overview_data.values() 
                                      if ai_info and ai_info.get("has_ai_overview", False))
            st.metric("Query con AI Overview", ai_overview_count)

        # Analisi AI Overview
        if ai_overview_count > 0:
            st.markdown("---")
            st.header("🤖 Analisi AI Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ai_percentage = (ai_overview_count / successful_queries) * 100 if successful_queries > 0 else 0
                st.metric("% Query con AI Overview", f"{ai_percentage:.1f}%")
                
                # Grafico AI Overview presence
                ai_presence_data = pd.DataFrame({
                    "Stato": ["Con AI Overview", "Senza AI Overview"],
                    "Count": [ai_overview_count, successful_queries - ai_overview_count]
                })
                
                if ai_presence_data["Count"].sum() > 0:
                    fig_ai = px.pie(
                        ai_presence_data,
                        values="Count",
                        names="Stato", 
                        title="Distribuzione AI Overview"
                    )
                    st.plotly_chart(fig_ai, use_container_width=True)
                else:
                    st.info("Nessun dato per il grafico AI Overview")
            
            with col2:
                # Top domini citati in AI Overview
                all_ai_domains = []
                for ai_info in ai_overview_data.values():
                    if ai_info and "ai_source_domains" in ai_info:
                        all_ai_domains.extend(ai_info["ai_source_domains"])
                
                if all_ai_domains:
                    ai_domains_counter = Counter(all_ai_domains)
                    ai_domains_df = pd.DataFrame(
                        ai_domains_counter.most_common(10),
                        columns=["Dominio", "Citazioni in AI Overview"]
                    )
                    
                    fig_ai_domains = px.bar(
                        ai_domains_df,
                        x="Citazioni in AI Overview",
                        y="Dominio",
                        orientation="h",
                        title="Top Domini Citati in AI Overview"
                    )
                    st.plotly_chart(fig_ai_domains, use_container_width=True)
                else:
                    st.info("Nessun dominio citato trovato negli AI Overview")
        else:
            st.info("ℹ️ Nessuna query ha attivato AI Overview per questa analisi.")

        # Grafici esistenti
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top Domini")
            if not domains_df.empty and len(domains_df) > 0:
                fig_domains = px.bar(
                    domains_df.head(10), 
                    x="Dominio", 
                    y="Occorrenze",
                    title="Top 10 Domini per Occorrenze"
                )
                fig_domains.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_domains, use_container_width=True)
            else:
                st.info("Nessun dominio trovato per creare il grafico")

        with col2:
            st.subheader("🏷️ Distribuzione Tipologie")
            if not page_type_df.empty and len(page_type_df) > 0:
                fig_pie = px.pie(
                    page_type_df, 
                    values="Occorrenze", 
                    names="Tipologia Pagina",
                    title="Tipologie di Pagine"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Nessuna tipologia di pagina trovata per creare il grafico")

        if keyword_clusters:
            st.subheader("🧠 Analisi Cluster Semantici")
            
            cluster_sizes = {name: len(keywords) for name, keywords in keyword_clusters.items()}
            if cluster_sizes:
                cluster_df = pd.DataFrame(list(cluster_sizes.items()), columns=["Cluster", "Numero Keyword"])
                
                fig_clusters = px.bar(
                    cluster_df.sort_values("Numero Keyword", ascending=False),
                    x="Cluster",
                    y="Numero Keyword", 
                    title="Distribuzione Keyword per Cluster"
                )
                fig_clusters.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_clusters, use_container_width=True)
            else:
                st.info("Nessun cluster creato per questa analisi")

        st.subheader("📋 Tabelle Dettagliate")
        
        tabs = ["Top Domini", "Tipologie Pagine", "Competitor Analysis", "AI Overview"]
        if keyword_clusters:
            tabs.append("Keyword Clustering")
        
        if len(tabs) == 5:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
        else:
            tab1, tab2, tab3, tab4 = st.tabs(tabs)
            tab5 = None
        
        with tab1:
            st.dataframe(domains_df, use_container_width=True)
        
        with tab2:
            st.dataframe(page_type_df, use_container_width=True)
        
        with tab3:
            st.dataframe(domain_page_types_df, use_container_width=True)
        
        with tab4:
            st.subheader("🤖 AI Overview per Query")
            if not ai_overview_df.empty:
                st.dataframe(ai_overview_df, use_container_width=True)
            
            st.subheader("📚 Fonti Citate in AI Overview")
            if not ai_sources_df.empty:
                st.dataframe(ai_sources_df, use_container_width=True)
        
        if tab5 and not clustering_df.empty:
            with tab5:
                st.dataframe(clustering_df, use_container_width=True)
                
                st.subheader("🔍 Dettagli Cluster")
                
                if custom_clusters:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Cluster Personalizzati:**")
                        personal_clusters = [k for k in keyword_clusters.keys() if k in custom_clusters and keyword_clusters[k]]
                        if personal_clusters:
                            selected_personal = st.selectbox(
                                "Seleziona cluster personalizzato:",
                                options=personal_clusters,
                                key="personal_cluster"
                            )
                        else:
                            st.write("Nessun cluster personalizzato con keyword")
                            selected_personal = None
                    
                    with col2:
                        st.write("**Cluster Automatici:**")
                        auto_clusters = [k for k in keyword_clusters.keys() if k not in custom_clusters]
                        if auto_clusters:
                            selected_auto = st.selectbox(
                                "Seleziona cluster automatico:",
                                options=auto_clusters,
                                key="auto_cluster"
                            )
                        else:
                            st.write("Nessun cluster automatico creato")
                            selected_auto = None
                    
                    selected_cluster = selected_personal or selected_auto
                else:
                    selected_cluster = st.selectbox(
                        "Seleziona un cluster per vedere i dettagli:",
                        options=list(keyword_clusters.keys())
                    )
                
                if selected_cluster and selected_cluster in keyword_clusters:
                    cluster_keywords = keyword_clusters[selected_cluster]
                    cluster_type = "Personalizzato" if selected_cluster in custom_clusters else "Automatico"
                    
                    st.write(f"**{selected_cluster}** ({cluster_type}) - {len(cluster_keywords)} keyword:")
                    
                    cols = st.columns(3)
                    for i, keyword in enumerate(cluster_keywords):
                        with cols[i % 3]:
                            st.write(f"• {keyword}")

        st.subheader("💾 Download Report")
        st.download_button(
            label="📥 Scarica Report Excel Completo",
            data=excel_data,
            file_name=f"serp_analysis_scrapingdog_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        progress_bar.empty()
        status_text.text("🎉 Analisi completata con successo!")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>SEO SERP Analyzer PRO con ScrapingDog - Analisi avanzata con AI Overview - Sviluppato con ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
