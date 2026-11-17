"""
corpus.py — Raw Roman Urdu sample sentences for development and testing.

Contract:
    Input:  (none — pure data module)
    Output: CORPUS: list[str]  — 50 raw, noisy Roman Urdu sentences.

These sentences intentionally contain:
- Emoji
- Repeated characters ("bohhht", "hahahaha")
- Mixed punctuation
- Chat abbreviations (mjy, tmhara, kya h)
- Code-mixed English words
- Spelling variants (acha/achha/accha)
"""

CORPUS: list[str] = [
    # Everyday greetings & feelings
    "Muje boht bhok lgi h 😂😂",
    "kya haal hai bhai??? sab theek thaak??",
    "mjy neend aa rhi h yaar 😴😴",
    "boht thak gaya hun aaj",
    "Kal school jana h subah subah uthna prega",
    # Food & hunger
    "bhook lagi hai kuch khaana chahiye",
    "aaj khana boht acha tha yar seriously!!",
    "pizza khaana chahta hun lekin paise nhi",
    "ghar pe kuch nhi khane ko 😭",
    "meri ammi ne aaj biryani banayi thi omg so goood",
    # Weather & environment
    "aaj garmi boht zyada h yarrr 🥵🥵🥵",
    "barish ho rhi hai baahar bahut tez",
    "thanda mausam accha lagta hai yar",
    "bijli chali gai fir se ugh",
    "aaj sooraj boht tez tha baahar jana mushkil tha",
    # School & studies
    "kal exam hai aur maine kuch nhi parha 😰",
    "math boht mushkil subject hai mjhe smjh nhi aata",
    "teacher ne homework diya hai boht zyada",
    "library mein baithke parhna acha lagta hai",
    "result aane wala hai darr lag raha hai",
    # Friends & social
    "yaar kahan ho tum kaafi dair se wait kar raha hun",
    "aaj dosto ke saath ghumne gaye the boht mazaa aaya",
    "party mein aana is weekend please please",
    "tmhara call aaya tha mene receive nhi kiya sorry",
    "kal milte hain chai peete hain baat karte hain",
    # Emotions
    "boht khushi ho rhi hai aaj ka din best tha",
    "dil udaas hai pata nhi kyun",
    "gussa aa raha hai baat hi mat karo",
    "aaj rona aaya boht zyada",
    "tension mat lo sab theek ho jaega",
    # Technology & phone
    "mera phone hang ho gaya hai fir se",
    "internet slow hai kuch load hi nhi ho raha",
    "naya phone lena chahta hun budget nhi hai",
    "app install karo ye boht kaam ki cheez hai",
    "battery khatam ho rhi thi charger nhi mila",
    # Travel & places
    "karachi ka traffic boht bura hai yaar seriously",
    "lahore gaya tha last month boht acha laga",
    "safar mein neend nhi aati akele",
    "ghar se bahar nikalna ache nhi lagta",
    "market jane ka mann hai kuch khareedna hai",
    # Mixed code / English words
    "meeting kal hai office mein prepare karo",
    "movie dekhi kya new wali? boht achi thi",
    "game khelna hai online koi hai khelne wala",
    "download ho gaya finally boht time laga",
    "project submit karna hai kal tak deadline hai",
    # Misc / expressive
    "hahahaha ye boht funny tha yaar omg 😂😂😂",
    "waah kya baat hai bilkul sahi kaha",
    "ufff itna kaam hai khatam hi nhi hota",
    "chalo theek hai kal dekhte hain",
    "Allah khair kare sab ke liye ameen",
]
