# loreal/ontology.py
'''
LOREAL_ONTOLOGY: A dictionary mapping beauty topics to their representative phrases and keywords.
    - Seeds: Representative phrases to feed into embeddings to capture the semantic space of a topic
    - Keywords: Exact text patterns to match directly in titles/descriptions/comments to catch literal mentions

GENERIC_YT: A set of generic YouTube topics that are down-weighted in scoring to avoid over-classification.
'''

LOREAL_ONTOLOGY= {
    "Makeup": {
        "seeds": [
            "makeup", "cosmetics", "foundation", "concealer", "mascara", "eyeliner",
            "lipstick", "lip gloss", "blush", "highlighter", "eyeshadow palette",
            "Maybelline", "Lancôme", "YSL Beauty", "L'Oréal Paris"
        ],
        "keywords": [
            "make up", "brow", "lip tint", "lip oil", "matte", "dewy", "smudgeproof",
            "yslbeauty", "ysl beauty", "maybelline", "loreal paris", "lancome"
        ]
    },
    "Skincare": {
        "seeds": [
            "skincare", "skin care routine", "cleanser", "toner", "serum", "moisturizer",
            "sunscreen", "SPF", "hydration", "acne", "anti-aging", "dark spots",
            "La Roche-Posay", "CeraVe", "Vichy", "Garnier", "L'Oréal Paris"
        ],
        "keywords": [
            "retinol", "hyaluronic acid", "niacinamide", "vitamin c", "salicylic acid",
            "AHA", "BHA", "glycolic acid", "microbiome", "ceramide", "sensitive skin",
            "la roche posay", "vichy", "garnier", "revitalift"
        ]
    },
    "Haircare": {
        "seeds": [
            "haircare", "shampoo", "conditioner", "hair mask", "hair oil", "heat protectant",
            "scalp care", "anti-dandruff", "Kérastase", "L'Oréal Professionnel", "Garnier Fructis"
        ],
        "keywords": [
            "kerastase", "loreal professionnel", "fructis", "split ends", "hair fall",
            "keratin", "leave-in", "balayage", "toning"
        ]
    },
    "Fragrance": {
        "seeds": [
            "fragrance", "perfume", "eau de parfum", "eau de toilette", "sillage",
            "fragrance notes", "YSL Libre", "Armani Si", "Maison Margiela REPLICA", "Lancôme La Vie Est Belle"
        ],
        "keywords": [
            "parfum", "EDP", "EDT", "notes", "accords", "projection", "longevity",
            "ysl libre", "la vie est belle", "black opium", "my way"
        ]
    },
    "Science/Ingredients": {
        "seeds": [
            "dermatology", "clinical study", "ingredient science", "polyphenols", "peptides",
            "retinoids", "SPF testing", "photoprotection"
        ],
        "keywords": [
            "in vitro", "in vivo", "dermatologist tested", "active ingredients",
            "non comedogenic", "hypoallergenic"
        ]
    },
    "Health & Wellness": {
        "seeds": [
            "health", "wellness", "healthy skin", "skin health", "nutrition", "lifestyle",
            "physical attractiveness", "mental health", "self care", "fitness", "exercise", "holistic"
        ],
        "keywords": [
            "health", "wellness", "healthy", "skin health", "nutrition", "lifestyle",
            "physical attractiveness", "mental health", "self care", "fitness", "exercise", "holistic"
        ]
    },
    "Fashion": {
        "seeds": [
            "fashion", "style", "outfit", "clothing", "runway", "trend", "designer", "couture"
        ],
        "keywords": [
            "fashion", "style", "outfit", "clothing", "runway", "trend", "designer", "couture"
        ]
    },
    "Nails": {
        "seeds": [
            "nail care", "manicure", "pedicure", "nail polish", "gel nails", "nail art",
            "cuticle care", "nail salon", "OPI", "Essie", "Sally Hansen"
        ],
        "keywords": [
            "nail polish", "gel nails", "manicure", "pedicure", "nail art", "cuticle care",
            "opi", "essie", "sally hansen", "nail salon"
        ]
    },
    "Men's Grooming": {
        "seeds": [
            "men's grooming", "beard care", "shaving", "hair styling", "men's skincare",
            "aftershave", "men's fragrance", "L'Oréal Men Expert", "Nivea Men", "Old Spice"
        ],
        "keywords": [
            "beard oil", "shaving cream", "men's skincare", "aftershave", "men's fragrance",
            "loreal men expert", "nivea men", "old spice"
        ]
    },
    "Sustainable Beauty": {
        "seeds": [
            "sustainable beauty", "eco-friendly", "cruelty-free", "organic", "natural ingredients",
            "green beauty", "zero waste", "recyclable packaging", "clean beauty"
        ],
        "keywords": [
            "eco-friendly", "cruelty-free", "organic", "natural ingredients", "green beauty",
            "zero waste", "recyclable packaging", "clean beauty"
        ]
    },
    "Beauty Tech": {
        "seeds": [
            "beauty tech", "smart skincare", "AI beauty", "skin analysis devices", "LED masks",
            "beauty apps", "virtual try-on", "3D printed beauty", "smart mirrors"
        ],
        "keywords": [
            "smart skincare", "AI beauty", "skin analysis devices", "LED masks", "beauty apps",
            "virtual try-on", "3D printed beauty", "smart mirrors"
        ]
    }
}

GENERIC_YT = {"Health", "Lifestyle (sociology)", "Entertainment", "Society"}