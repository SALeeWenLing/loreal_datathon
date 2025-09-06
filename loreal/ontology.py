# loreal/ontology.py
'''
LOREAL_ONTOLOGY: A dictionary mapping beauty topics to their representative phrases and keywords.
    - Seeds: Representative phrases to feed into embeddings to capture the semantic space of a topic
    - Keywords: Exact text patterns to match directly in titles/descriptions/comments to catch literal mentions

GENERIC_YT: A set of generic YouTube topics that are down-weighted in scoring to avoid over-classification.
'''

LOREAL_ONTOLOGY = {
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
    }
}
GENERIC_YT = {"Health", "Lifestyle (sociology)", "Entertainment", "Society"}
