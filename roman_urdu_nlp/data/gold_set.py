"""
gold_set.py — Hand-crafted gold-standard evaluation set for the roman_urdu_nlp pipeline.

Contract:
    Input:  (none — pure data module)
    Output: GOLD_SET: list[dict]
        Each dict has keys:
            "raw"        : str   — raw noisy input (as a user would type)
            "normalized" : str   — expected canonical Roman Urdu after cleaning + normalization
            "urdu"       : str   — expected Urdu script transliteration
            "english"    : str   — expected English translation

Assumptions:
  - Normalized form uses canonical spellings from the normalization dictionary.
  - Urdu script follows standard written Urdu orthography.
  - English translation is natural, not word-for-word.
  - Emojis and punctuation noise are NOT present in normalized/urdu/english fields.
"""

GOLD_SET: list[dict] = [
    {
        "raw": "Muje boht bhok lgi h 😂😂",
        "normalized": "Mujhe bohat bhook lagi hai",
        "urdu": "مجھے بہت بھوک لگی ہے",
        "english": "I am very hungry.",
    },
    {
        "raw": "kya haal hai bhai??? sab theek thaak??",
        "normalized": "kya haal hai bhai sab theek thaak",
        "urdu": "کیا حال ہے بھائی سب ٹھیک ٹھاک",
        "english": "How are you brother? Is everything alright?",
    },
    {
        "raw": "mjy neend aa rhi h yaar 😴😴",
        "normalized": "mujhe neend aa rahi hai yaar",
        "urdu": "مجھے نیند آ رہی ہے یار",
        "english": "I am feeling sleepy, friend.",
    },
    {
        "raw": "boht thak gaya hun aaj",
        "normalized": "bohat thak gaya hoon aaj",
        "urdu": "بہت تھک گیا ہوں آج",
        "english": "I am very tired today.",
    },
    {
        "raw": "aaj garmi boht zyada h yarrr 🥵🥵🥵",
        "normalized": "aaj garmi bohat zyada hai yaar",
        "urdu": "آج گرمی بہت زیادہ ہے یار",
        "english": "It is very hot today, friend.",
    },
    {
        "raw": "kal exam hai aur maine kuch nhi parha 😰",
        "normalized": "kal exam hai aur maine kuch nahi parha",
        "urdu": "کل امتحان ہے اور میں نے کچھ نہیں پڑھا",
        "english": "There is an exam tomorrow and I have not studied anything.",
    },
    {
        "raw": "aaj khana boht acha tha yar seriously!!",
        "normalized": "aaj khana bohat acha tha yaar",
        "urdu": "آج کھانا بہت اچھا تھا یار",
        "english": "The food was very good today, friend.",
    },
    {
        "raw": "barish ho rhi hai baahar bahut tez",
        "normalized": "barish ho rahi hai baahar bahut tez",
        "urdu": "بارش ہو رہی ہے باہر بہت تیز",
        "english": "It is raining heavily outside.",
    },
    {
        "raw": "yaar kahan ho tum kaafi dair se wait kar raha hun",
        "normalized": "yaar kahan ho tum kaafi dair se wait kar raha hoon",
        "urdu": "یار کہاں ہو تم کافی دیر سے انتظار کر رہا ہوں",
        "english": "Friend, where are you? I have been waiting for a long time.",
    },
    {
        "raw": "mera phone hang ho gaya hai fir se",
        "normalized": "mera phone hang ho gaya hai phir se",
        "urdu": "میرا فون ہینگ ہو گیا ہے پھر سے",
        "english": "My phone has hung again.",
    },
    {
        "raw": "dil udaas hai pata nhi kyun",
        "normalized": "dil udaas hai pata nahi kyun",
        "urdu": "دل اداس ہے پتہ نہیں کیوں",
        "english": "The heart is sad, do not know why.",
    },
    {
        "raw": "tension mat lo sab theek ho jaega",
        "normalized": "tension mat lo sab theek ho jaega",
        "urdu": "ٹینشن مت لو سب ٹھیک ہو جائے گا",
        "english": "Do not worry, everything will be fine.",
    },
    {
        "raw": "karachi ka traffic boht bura hai yaar seriously",
        "normalized": "karachi ka traffic bohat bura hai yaar",
        "urdu": "کراچی کا ٹریفک بہت برا ہے یار",
        "english": "Karachi's traffic is very bad, friend.",
    },
    {
        "raw": "hahahaha ye boht funny tha yaar omg 😂😂😂",
        "normalized": "haha ye bohat funny tha yaar",
        "urdu": "ہاہا یہ بہت مضحکہ خیز تھا یار",
        "english": "Haha, this was very funny, friend.",
    },
    {
        "raw": "kal milte hain chai peete hain baat karte hain",
        "normalized": "kal milte hain chai peete hain baat karte hain",
        "urdu": "کل ملتے ہیں چائے پیتے ہیں بات کرتے ہیں",
        "english": "Let us meet tomorrow, drink tea, and have a talk.",
    },
    {
        "raw": "math boht mushkil subject hai mjhe smjh nhi aata",
        "normalized": "math bohat mushkil subject hai mujhe samajh nahi aata",
        "urdu": "ریاضی بہت مشکل مضمون ہے مجھے سمجھ نہیں آتا",
        "english": "Math is a very difficult subject, I do not understand it.",
    },
    {
        "raw": "Allah khair kare sab ke liye ameen",
        "normalized": "Allah khair kare sab ke liye ameen",
        "urdu": "اللہ خیر کرے سب کے لیے آمین",
        "english": "May Allah grant goodness to everyone, Ameen.",
    },
    {
        "raw": "boht khushi ho rhi hai aaj ka din best tha",
        "normalized": "bohat khushi ho rahi hai aaj ka din best tha",
        "urdu": "بہت خوشی ہو رہی ہے آج کا دن بہترین تھا",
        "english": "I am very happy, today was the best day.",
    },
    {
        "raw": "internet slow hai kuch load hi nhi ho raha",
        "normalized": "internet slow hai kuch load hi nahi ho raha",
        "urdu": "انٹرنیٹ سست ہے کچھ لوڈ ہی نہیں ہو رہا",
        "english": "The internet is slow, nothing is loading.",
    },
    {
        "raw": "project submit karna hai kal tak deadline hai",
        "normalized": "project submit karna hai kal tak deadline hai",
        "urdu": "پروجیکٹ جمع کرنا ہے کل تک ڈیڈلائن ہے",
        "english": "I have to submit the project, the deadline is tomorrow.",
    },
]
