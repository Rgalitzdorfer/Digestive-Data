#Import Necessary Libraries
import time, re, requests, pandas as pd, nltk
from pathlib import Path
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import itertools
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

#Configure 
SUBS = ["ibs", "CrohnsDisease", "UlcerativeColitis", "microbiome"]  
PER_SUB_LIMIT = 800        
SLEEP_BETWEEN_PAGES = 1.5    
UA = {"User-Agent": "DigestiveDataBot/0.1 by u/DigestiveData"}  

#Output Folder Setup
OUT = Path("data"); OUT.mkdir(exist_ok=True)

#Fetch From Reddit Public JSON
def fetch_sub(sub, limit=100):
    url = f"https://www.reddit.com/r/{sub}/new.json"
    params = {"limit": 100}
    posts, after, got = [], None, 0
    while got < limit:
        if after: params["after"] = after
        r = requests.get(url, headers=UA, params=params, timeout=20)
        if r.status_code != 200:
            print(f"r/{sub} HTTP {r.status_code}; stopping.")
            break
        j = r.json()
        children = j.get("data", {}).get("children", [])
        if not children:
            break
        for c in children:
            d = c["data"]
            posts.append({
                "subreddit": sub,
                "id": d.get("id"),
                "created_utc": d.get("created_utc"),
                "title": d.get("title", ""),
                "selftext": d.get("selftext", ""),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "permalink": "https://reddit.com"+d.get("permalink",""),
            })
        got += len(children)
        after = j["data"].get("after")
        if not after: break
        time.sleep(SLEEP_BETWEEN_PAGES)
    print(f"r/{sub}: {len(posts)} posts")
    return posts

#Run Fetch Across Subreddits
all_rows = []
for s in SUBS:
    all_rows += fetch_sub(s, PER_SUB_LIMIT)

#Save Raw Data From Reddit
df = pd.DataFrame(all_rows).drop_duplicates("id")
df.to_csv(OUT/"posts_raw.csv", index=False)
print("Saved raw:", len(df), OUT/"posts_raw.csv")

#Clean & Normalize Text
nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
def clean(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(w for w in t.split() if len(w) > 2 and w not in STOP)
df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).map(clean)
df = df[df["text"].str.len() > 0]
df.to_csv(OUT/"posts_clean.csv", index=False)
print("Saved clean:", len(df), OUT/"posts_clean.csv")

#Generate Wordcloud From Corpus 
blob = " ".join(df["text"].tolist())
wc = WordCloud(width=1200, height=600, background_color="white").generate(blob)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off"); plt.tight_layout()
plt.savefig("wordcloud.png", dpi=200)
print("Saved wordcloud.png")


#Define Dictionaries For Matching
#Symptoms
SYMPTOM_SINGLE = {
    "pain","cramp","cramps","bloat","bloating","diarrhea","constipation","nausea","vomit","vomiting",
    "flare","flareup","gas","urgency","bleeding","fatigue","tired","exhaustion","anxiety","depression",
    "stress","mucus","fever","cramps","cramping","spasm","ache","indigestion","heartburn","reflux",
    "dizziness","bloating","constipated"
}
SYMPTOM_PHRASES = {
    "brain fog","weight loss","stomach pain","abdominal pain","lower abdominal pain","chest pain"
}
#Triggers
TRIGGER_SINGLE = {
    "dairy","milk","lactose","gluten","wheat","spicy","greasy","fatty","coffee","caffeine","alcohol",
    "beer","wine","sugar","sweeteners","fructose","onion","garlic","fiber","fried","raw","nuts",
    "seeds","broccoli","cauliflower","beans","legumes","probiotic","probiotics","antibiotic",
    "antibiotics","ibuprofen","nsaid","nsaids","stress","anxiety","heat","travel","vacation","exercise",
    "sleep","hormones","period","menstruation","kefir","yogurt","kimchi","sauerkraut"
}
TRIGGER_PHRASES = {
    "low fodmap","high fiber","artificial sweetener","artificial sweeteners","red meat","fast food",
    "energy drink","energy drinks","carbonated drinks","intermittent fasting"
}

#Compile Phrase Regexes
def phrase_re(phrase):
    # turn "brain fog" -> r"\bbrain\s+fog\b"
    p = r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"
    return re.compile(p)

SYMPTOM_PHRASE_RES = [phrase_re(p) for p in SYMPTOM_PHRASES]
TRIGGER_PHRASE_RES = [phrase_re(p) for p in TRIGGER_PHRASES]

#Tokenize Helper
def to_tokens(t):
    return t.split()  

#Match Terms In Text
def match_terms(text):
    tokens = set(to_tokens(text))
    found_sym = set(SYMPTOM_SINGLE) & tokens
    found_trg = set(TRIGGER_SINGLE) & tokens

    for rx, label in zip(SYMPTOM_PHRASE_RES, SYMPTOM_PHRASES):
        if rx.search(text):
            found_sym.add(label)
    for rx, label in zip(TRIGGER_PHRASE_RES, TRIGGER_PHRASES):
        if rx.search(text):
            found_trg.add(label)

    return found_sym, found_trg

#Apply Matching Across Corpus
sym_list, trg_list = [], []
for t in df["text"].astype(str):
    s, g = match_terms(t)
    sym_list.append(sorted(s))
    trg_list.append(sorted(g))
df["symptoms_found"] = sym_list
df["triggers_found"] = trg_list

#Save Tagged Dataset
df.to_csv("data/posts_tagged.csv", index=False)

#Get Count of All Unique Symptoms & Triggers
def explode_and_count(series_of_lists):
    c = Counter(itertools.chain.from_iterable(series_of_lists))
    return pd.DataFrame({"term": list(c.keys()), "count": list(c.values())}).sort_values("count", ascending=False)
sym_freq_all = explode_and_count(df["symptoms_found"])
trg_freq_all = explode_and_count(df["triggers_found"])
sym_freq_all.to_csv("data/symptom_counts_overall.csv", index=False)
trg_freq_all.to_csv("data/trigger_counts_overall.csv", index=False)

#Per-Subreddit Counts
sym_per_sub, trg_per_sub = [], []
for sub, dfx in df.groupby("subreddit"):
    sf = explode_and_count(dfx["symptoms_found"]); sf["subreddit"] = sub
    tf = explode_and_count(dfx["triggers_found"]); tf["subreddit"] = sub
    sym_per_sub.append(sf); trg_per_sub.append(tf)
sym_per_sub = pd.concat(sym_per_sub, ignore_index=True)
trg_per_sub = pd.concat(trg_per_sub, ignore_index=True)
sym_per_sub.to_csv("data/symptom_counts_by_subreddit.csv", index=False)
trg_per_sub.to_csv("data/trigger_counts_by_subreddit.csv", index=False)

#Symptom & Trigger Cooccurrence
co_counts = defaultdict(int)
for s_terms, g_terms in zip(df["symptoms_found"], df["triggers_found"]):
    for s in s_terms:
        for g in g_terms:
            co_counts[(s, g)] += 1

co_df = pd.DataFrame([(s, g, c) for (s, g), c in co_counts.items()],
                     columns=["symptom","trigger","count"]).sort_values("count", ascending=False)
co_df.to_csv("data/symptom_trigger_cooccurrence.csv", index=False)

#Top 15 Symptoms & Triggers (Use For Refining)
def bar_top(df_counts, title, outfile, topn=15):
    top = df_counts.head(topn)[::-1]  
    plt.figure(figsize=(8, 6))
    plt.barh(top["term"], top["count"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
bar_top(sym_freq_all, "Top Symptoms Mentioned (All Subs)", "top_symptoms.png")
bar_top(trg_freq_all, "Top Triggers Mentioned (All Subs)", "top_triggers.png")

#Print To Validate
print("Saved:")
print("- data/posts_tagged.csv")
print("- data/symptom_counts_overall.csv")
print("- data/trigger_counts_overall.csv")
print("- data/symptom_counts_by_subreddit.csv")
print("- data/trigger_counts_by_subreddit.csv")
print("- data/symptom_trigger_cooccurrence.csv")
print("- top_symptoms.png")
print("- top_triggers.png")


#FEATURE ENGINEERING: Make Useful Groups Out of Raw Symptoms & Triggers
def map_symptom(term: str) -> str:
    t = term.lower()
    #Pain & Cramping
    if t in {"pain","ache","cramp","cramps","cramping","spasm","stomach pain","abdominal pain","lower abdominal pain","chest pain"}:
        return "Pain/Cramping"
    #Bowel Changes & Bleeding
    if t in {"diarrhea","constipation","constipated","urgency","bleeding","mucus","flare","flareup"}:
        return "Bowel"
    #Bloating & Gas
    if t in {"bloat","bloating","gas"}:
        return "Bloating"
    #Upper GI & Nausea
    if t in {"nausea","vomit","vomiting","indigestion","heartburn","reflux"}:
        return "Nausea/Reflux"
    #Systemic & Mood
    if t in {"fatigue","tired","exhaustion","brain fog","fever","dizziness","anxiety","depression","stress","weight loss"}:
        return "Lifestyle/External"
    return "Other/Minor"
sym_groups = sym_per_sub.copy()
sym_groups["group"] = sym_groups["term"].map(map_symptom)

#Keep 5 For Main Buckets
sym_groups = sym_groups[sym_groups["group"] != "Other/Minor"]

#Aggregate Counts Across Each Subreddit By Group
sym_g = (sym_groups
         .groupby(["group","subreddit"], as_index=False)["count"].sum())
sym_g["pct"] = 100 * sym_g["count"] / sym_g.groupby("group")["count"].transform("sum") #Percentage Breakdown
sym_g_piv = sym_g.pivot_table(index="group", columns="subreddit", values="pct", fill_value=0) #Stacked Bar
order = (sym_groups.groupby("group")["count"].sum() #Order Rows
         .sort_values(ascending=False).index.tolist())
sym_g_piv = sym_g_piv.loc[order]

#Consistent Colors
sub_colors = {
    "ibs": "#1f77b4",
    "CrohnsDisease": "#ff7f0e",
    "UlcerativeColitis": "#2ca02c",
    "microbiome": "#9467bd", #Not Used
}

#Plot Stacked Bar For Symptoms
plt.figure(figsize=(9, 5))
sym_g_piv.plot(kind="bar", stacked=True,
               color=[sub_colors.get(c, "#666") for c in sym_g_piv.columns])
plt.ylabel("Share of Mentions (%)")
plt.title("Symptom Buckets — Percentage Breakdown by Subreddit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("symptom_groups_stacked.png", dpi=200)
plt.close()
print("Saved symptom_groups_stacked.png")


#FEATURE ENGINEERING: Make Useful Groups For Triggers
def map_trigger(term: str) -> str:
    t = term.lower()
    #Dairy & Lactose
    if t in {"dairy","milk","lactose","yogurt","kefir"}:
        return "Dairy"
    #Gluten & Grains
    if t in {"gluten","wheat"}:
        return "Gluten"
    #FODMAP & Rich Foods 
    if t in {"onion","garlic","beans","legumes","broccoli","cauliflower","fiber",
             "spicy","greasy","fatty","fried","nuts","seeds","high fiber","low fodmap"}:
        return "FODMAP/Rich Foods"
    #Beverages & Sweeteners
    if t in {"coffee","caffeine","alcohol","beer","wine","energy drink","energy drinks",
             "carbonated drinks","sugar","sweeteners","fructose"}:
        return "Drinks/Sweets"
    #Lifestyle, Meds & Microbiome
    if t in {"stress","anxiety","sleep","heat","travel","vacation","exercise","hormones",
             "period","menstruation","antibiotic","antibiotics","ibuprofen","nsaid","nsaids",
             "probiotic","probiotics","kimchi","sauerkraut"}:
        return "Lifestyle/Medication"
    return "Other/Minor"

#Use For Grouping & Stacked Bar Plot
trg_groups = trg_per_sub.copy()
trg_groups["group"] = trg_groups["term"].map(map_trigger)
trg_groups = trg_groups[trg_groups["group"] != "Other/Minor"]
trg_g = (trg_groups
         .groupby(["group","subreddit"], as_index=False)["count"].sum())
trg_g["pct"] = 100 * trg_g["count"] / trg_g.groupby("group")["count"].transform("sum")

trg_g_piv = trg_g.pivot_table(index="group", columns="subreddit", values="pct", fill_value=0)

order_t = (trg_groups.groupby("group")["count"].sum()
           .sort_values(ascending=False).index.tolist())
trg_g_piv = trg_g_piv.loc[order_t]


#Keep Consistent Subreddit View
target_subs = ["CrohnsDisease", "UlcerativeColitis"]

#For Symptoms
sym_g_piv = sym_g_piv.reindex(columns=target_subs, fill_value=0)
sym_g_piv = sym_g_piv.div(sym_g_piv.sum(axis=1).replace(0, 1), axis=0) * 100
ax = sym_g_piv.plot(kind="bar", stacked=True,
                    color=[sub_colors[s] for s in target_subs],
                    figsize=(9, 5))
plt.ylabel("Share of Mentions (%)")
plt.title("Symptom Buckets — Percentage Breakdown by Subreddit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("symptom_groups_stacked.png", dpi=200)
plt.close()

#For Triggers
trg_g_piv = trg_g_piv.reindex(columns=target_subs, fill_value=0)
trg_g_piv = trg_g_piv.div(trg_g_piv.sum(axis=1).replace(0, 1), axis=0) * 100
ax = trg_g_piv.plot(kind="bar", stacked=True,
                    color=[sub_colors[s] for s in target_subs],
                    figsize=(9, 5))
plt.ylabel("Share of Mentions (%)")
plt.title("Trigger Buckets — Percentage Breakdown by Subreddit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("trigger_groups_stacked.png", dpi=200)
plt.close()
print("Saved trigger_groups_stacked.png")

#Crohns Symptoms & Trigger For Comparison
crohns_sym = (sym_per_sub[sym_per_sub["subreddit"]=="CrohnsDisease"]
              .sort_values("count", ascending=False)
              .head(5))

crohns_trg = (trg_per_sub[trg_per_sub["subreddit"]=="CrohnsDisease"]
              .sort_values("count", ascending=False)
              .head(5))

#UC Symptoms & Triggers For Comparison
uc_sym = (sym_per_sub[sym_per_sub["subreddit"]=="UlcerativeColitis"]
          .sort_values("count", ascending=False)
          .head(5))

uc_trg = (trg_per_sub[trg_per_sub["subreddit"]=="UlcerativeColitis"]
          .sort_values("count", ascending=False)
          .head(5))

##PLOT
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes[0,0].barh(crohns_sym["term"], crohns_sym["count"])
axes[0,0].set_title("Crohn’s – Top Symptoms")
axes[0,1].barh(crohns_trg["term"], crohns_trg["count"])
axes[0,1].set_title("Crohn’s – Top Triggers")
axes[1,0].barh(uc_sym["term"], uc_sym["count"])
axes[1,0].set_title("UC – Top Symptoms")
axes[1,1].barh(uc_trg["term"], uc_trg["count"])
axes[1,1].set_title("UC – Top Triggers")
#Format
plt.tight_layout()
plt.savefig("top5_symptoms_triggers_crohns_uc.png", dpi=200)
plt.close()


