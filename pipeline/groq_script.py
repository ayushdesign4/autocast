"""Groq LLM integration — generate Short script + image prompts as structured JSON.

Generates Devanagari Hindi narration (130-180 words) and English visual prompts for vertical 9:16 Shorts.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from groq import Groq

from pipeline.channel_presets import ChannelPreset
from pipeline.story_history import history_prompt_block

log = logging.getLogger("pipeline.groq_script")

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
BACKUP_GROQ_MODEL = "llama-3.1-8b-instant"

FALLBACK_CATALOG: list[dict[str, Any]] = [
    {
        "topic": "गर्मियों की शाम को छत पर पानी का छिड़काव और चारपाई पर तारों भरी रात",
        "youtube_title": "छत पर पानी का छिड़काव और तारों भरी रात की यादें",
        "youtube_description": (
            "1990 के दौर में गर्मियों की रातें छत पर चारपाई और तारों के साए में कैसे कटती थीं। "
            "क्या आपको भी याद है दादी का हाथ का पंखा? कमेंट्स में जरूर बताएं! "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "गर्मियों की छुट्टियों में शाम ढलते ही छत पर पाइप से पानी का छिड़काव करने की वो सौंधी खुशबू आज भी याद आती है। "
            "लाल गर्म ईंटों से उठती भाप धीरे-धीरे ठंडी हवा में बदल जाती थी और पूरा परिवार मिलकर छत पर चारपाइयां बिछाता था। "
            "सफेद सूती चादरें झाड़कर बिछाई जाती थीं और दादी का हाथ का बना पंखा धीरे-धीरे चलता था। "
            "रात को खुले आसमान के नीचे लेटकर तारों को गिनना और दादाजी से राजा-रानी और परियों की कहानियां सुनना, नींद कब आ जाती थी पता ही नहीं चलता था। "
            "सुबह चिड़ियों की चहचहाहट और चेहरे पर पड़ती पहली धूप से आंख खुलती थी। "
            "न कोई एसी की बनावटी ठंडक थी और न पंखे की तेज आवाज, बस तारों भरी ठंडी रात, अपनों का प्यार और बेफिक्र सुकून की वो मीठी नींद थी जो आज के महंगे गद्दों पर भी नहीं मिलती।"
        ),
        "image_prompts": [
            "An Indian brick rooftop in late evening golden hour, a young boy holding a green water hose spraying water on red terracotta floor tiles with mist rising, vertical 9:16 portrait",
            "Traditional Indian wooden woven charpai beds neatly arranged on a wide terrace, crisp white cotton sheets being laid out by a mother, sunset skyline, vertical 9:16",
            "An affectionate elderly grandmother sitting on a charpai gently fanning a child with a traditional hand-woven bamboo hand fan, twilight sky, vertical 9:16",
            "A grandfather lying under the starry night sky pointing upwards at constellations to fascinated grandchildren, rustic terrace parapet, vertical 9:16",
            "Wide angle view of a peaceful Indian village and small town rooftops under thousands of sparkling stars and soft moonlight, quiet serene atmosphere, vertical 9:16",
            "Early morning sunrise breaking over an Indian terrace, gentle golden rays falling on sleeping family members wrapped in thin cotton quilts, sparrows on wall, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "indian nostalgia", "summer memories", "terrace nights", "childhood", "shorts"],
    },
    {
        "topic": "किराए की वीसीआर कैसेट, उलझी रील और शुक्रवार की रात मोहल्ला सिनेमा",
        "youtube_title": "किराए का VCR और मोहल्ले का वीकेंड सिनेमा",
        "youtube_description": (
            "जब शुक्रवार को किराए पर VCR आता था और पूरा मोहल्ला एक कमरे में सिमट जाता था। "
            "पेंसिल से कैसेट की रील घुमाने का वो जमाना याद है? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "1990 के दौर में शनिवार की शाम का इंतजार हर घर को बेसब्री से रहता था, जब वीडियो लाइब्रेरी से किराए पर वीसीआर और दो वीडियो कैसेट लाए जाते थे। "
            "नीले प्लास्टिक के डिब्बे में रखी वो बड़ी सी कैसेट, जिसे पेंसिल डालकर बड़े जतन से उल्टा घुमाया जाता था ताकि रील उलझ न जाए। "
            "शाम होते ही मोहल्ले के सारे बच्चे और बड़े एक ही कमरे में दरी पर बैठ जाते थे। "
            "जब टीवी स्क्रीन पर नीली पट्टी आती और वीसीआर का हेड घूमने की आवाज गूंजती, तो सबकी सांसें थम जाती थीं। "
            "बीच में जब पिक्चर हिलने लगती, तो कोई उठकर ट्रैकिंग का बटन घुमाता था। "
            "साथ में गरमा-गरम पकौड़े और चाय का दौर आधी रात तक चलता रहता था। "
            "उस दौर में सिनेमा सिर्फ पर्दे पर नहीं, पूरे मोहल्ले के दिल में एक साथ उतरता था।"
        ),
        "image_prompts": [
            "A retro 1990s small Indian video cassette rental shop lined with colorful VHS tape boxes, a teenager eagerly renting a VCR player, warm fluorescent lighting, vertical 9:16",
            "Close-up of hands inserting a yellow Natraj pencil into the spool hole of a black VHS magnetic cassette tape to rewind the reel, vertical 9:16 portrait",
            "A crowded Indian living room packed with neighbors and children sitting cross-legged on a patterned durrie rug watching a glowing CRT TV, vertical 9:16",
            "Front view of a vintage silver VCR deck connected to an Onida television, tape loader mechanism glowing softly, evening setting, vertical 9:16",
            "A mother serving hot steaming pakodas and steel glasses of spiced chai to smiling guests watching an old Hindi movie, vertical 9:16",
            "Late night emotional atmosphere, tired children asleep against their parents' shoulders while the television screen casts blue flicker into the quiet room, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "vcr memories", "retro bollywood", "indian childhood", "weekend cinema", "shorts"],
    },
    {
        "topic": "स्कूल की घंटी, पहली बारिश और कागज़ की नाव तैराने का रोमांच",
        "youtube_title": "बारिश का पहला दिन, स्कूल की घंटी और कागज़ की नाव",
        "youtube_description": (
            "स्कूल की छुट्टी की घंटी और बारिश में कागज़ की नाव बहाने की वो बेफिक्र खुशियां। "
            "क्या आपकी नाव भी उस पानी के बहाव में तैरती थी? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "सावन की पहली झमाझम बारिश और स्कूल की आखिरी घंटी बजने का वो रोमांच आज के बच्चों को कौन समझाए। "
            "जैसे ही छुट्टी की घंटी टन-टन बजती, बस्तों को सिर पर रखकर सारे बच्चे बारिश में भीगते हुए सड़क पर दौड़ पड़ते थे। "
            "रास्ते में बहते मटमैले पानी के नालों में पुरानी कॉपी के पन्नों से बनी रंग-बिरंगी कागज़ की नावें तैराना किसी बड़े मिशन से कम नहीं था। "
            "अगर किसी की नाव दूर तक बिना डूबे तैर जाती, तो पूरे मोहल्ले में खुशी की तालियां बज उठती थीं। "
            "घर पहुंचते ही मां का डांटना और फिर प्यार से तौलिये से सिर पोंछकर गरमा-गरम चाय और बेसन के कुरकुरे पकोड़े थमा देना, वो ममता अनमोल थी। "
            "वो भीगे हुए मोज़े, किताबों की भीनी-भीनी खुशबू और वो बचपन की भोली सी नाव आज यादों के दरिया में तैर रही है।"
        ),
        "image_prompts": [
            "An Indian government school courtyard under heavy pouring monsoon rain, an iron bell hanging on an archway ringing, children cheering, vertical 9:16",
            "Schoolchildren in wet blue-and-white uniforms running happily down a rain-soaked lane holding school bags over their heads, puddles splashing, vertical 9:16",
            "Close-up of children's muddy hands placing colorful folded paper boats into a rushing stream of rainwater along the street edge, vertical 9:16",
            "A delicate white paper boat floating gracefully on rain water ripples, reflecting the cloudy overcast sky, vertical 9:16 portrait",
            "A caring Indian mother at the wooden front doorway drying her smiling son's dripping wet hair with a yellow cotton towel, warm kitchen light behind, vertical 9:16",
            "A plate of fresh crispy golden besan pakodas next to a glass of cutting chai on a window sill as raindrops streak down the glass pane, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "monsoon memories", "paper boats", "school days", "rainy childhood", "shorts"],
    },
    {
        "topic": "मोहल्ले में बर्फ का गोला और खट्टा-मीठा काला खट्टा का स्वाद",
        "youtube_title": "बर्फ का गोला और बचपन की वो खट्टी-मीठी शामें",
        "youtube_description": (
            "दो रुपये का सिक्का और बर्फ के गोले पर काला खट्टा की वो जादुई चाशनी। "
            "किस-किस को याद है अपनी रंग-बिरंगी जीभ दिखाना? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "तपती दोपहरी में जब मोहल्ले की गली में बर्फ के गोले वाले ठेले की घंटी बजती थी, तो नींद से भरी आंखें भी एक झटके में खुल जाती थीं। "
            "हाथ में दो रुपये का सिक्का थामे नंगे पांव दौड़कर ठेले के सामने खड़े हो जाना बचपन का सबसे बड़ा सुख था। "
            "बर्फ की सिल्ली पर लोहे का रंदा चलने की वो चिर्र-चिर्र आवाज और फिर बर्फ के गोले पर खट्टा-मीठा काला खट्टा, लाल गुलाब और हरी खस की चाशनी का डलना किसी जादू जैसा लगता था। "
            "गोला चूसते-चूसते जब जीभ पूरी लाल या काली हो जाती, तो हम एक-दूसरे को जीभ दिखाकर जोर-जोर से हंसते थे। "
            "आज बड़े-बड़े मॉल में महंगी आइसक्रीम तो मिल जाती है, लेकिन उस बर्फ के गोले में घुली हुई बचपन की वो मीठी ठंडक और मासूम खिलखिलाहट अब कहीं नजर नहीं आती।"
        ),
        "image_prompts": [
            "A colorful vintage wooden handcart parked in a narrow Indian street with large crystal clear ice block wrapped in wet jute sack, brass bell ringing, vertical 9:16",
            "The gola vendor shaving crushed snowy ice using a traditional heavy iron hand plane, ice shavings flying under afternoon sunlight, vertical 9:16 portrait",
            "Rows of bright glass syrup bottles filled with dark purple kala khatta, crimson rose, and green khas syrups glowing in sunlight on the wooden cart, vertical 9:16",
            "Close-up of a perfectly formed crushed ice gola on a bamboo stick drenched in multi-colored sweet tangy syrups, vertical 9:16",
            "Three Indian children laughing hysterically and sticking out their brightly stained purple and red tongues to each other in a sunny street corner, vertical 9:16",
            "The gola cart slowly moving away down the tranquil tree-lined neighborhood as evening shadows stretch, warm nostalgic atmosphere, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "barf ka gola", "street food", "childhood summer", "kala khatta", "shorts"],
    },
    {
        "topic": "रेडियो पर बिनाका गीतमाला और अमीन सयानी की जादुई आवाज",
        "youtube_title": "रेडियो पर बिनाका गीतमाला और अमीन सयानी की आवाज",
        "youtube_description": (
            "बुधवार रात आठ बजे रेडियो सीलोन पर अमीन सयानी की वो खनकती आवाज: 'बहनों और भाइयों'। "
            "क्या आप भी डायरी में गानों की रैंकिंग लिखते थे? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "बुधवार की शाम को ठीक आठ बजते ही घर के लकड़ी वाले रेडियो के सामने पूरा परिवार सांस रोककर बैठ जाता था। "
            "जैसे ही रेडियो सीलोन से अमीन सयानी की खनकती हुई आवाज गूंजती, 'बहनों और भाइयों', पूरे कमरे में एक जादुई खामोशी छा जाती थी। "
            "बिनाका गीतमाला का वो सुरीला सफर और पायदान पर चढ़ते नए गानों का रोमांच हर दिल की धड़कन बढ़ा देता था। "
            "हम सब हाथ में डायरी और कलम लिए बैठते थे ताकि अपने पसंदीदा गाने की रैंकिंग लिख सकें। "
            "रेडियो की सुई को आहिस्ता-आहिस्ता घुमाकर स्टेशन पकड़ना और खरखराहट के बीच भी संगीत की मधुर तान को महसूस करना एक अलग ही नशा था। "
            "आज उंगलियों पर लाखों गाने मौजूद हैं, पर उस एक घंटे के रेडियो कार्यक्रम में जो पूरे परिवार का जुड़ाव था, वो अब खो गया है।"
        ),
        "image_prompts": [
            "A vintage wooden Bush valve radio set placed on a lace-covered table, needle dial illuminated with soft amber light, vertical 9:16 portrait",
            "A family sitting together in a cozy 1990s living room listening intently to the radio broadcast, warm ambient room light, vertical 9:16",
            "Close-up of a schoolgirl's hand writing Hindi song rankings and titles in an old school notebook with a blue Reynolds ballpoint pen, vertical 9:16",
            "Gentle macro shot of someone's fingers delicately tuning the brass knob of the retro radio, needle moving across shortwave frequency lines, vertical 9:16",
            "A smiling father and teenage son exchanging an excited glance as the countdown reveals the number one hit song of the week, vertical 9:16",
            "A quiet evening window looking out into the twilight sky while the golden radio glow warms the wooden room, peaceful nostalgia, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "binaca geetmala", "radio ceylon", "vintage radio", "retro music", "shorts"],
    },
    {
        "topic": "डाकिया बाबू की घंटी और नीले इनलैंड लेटर की खुशबू",
        "youtube_title": "डाकिया बाबू की साइकिल और नीले इनलैंड लेटर का वो दौर",
        "youtube_description": (
            "जब डाकिया बाबू की साइकिल की घंटी बजती थी और नीले इनलैंड लेटर में पूरे परिवार का हाल आता था। "
            "हाथ से लिखी चिट्ठियों का वो अहसास याद है? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "मोहल्ले की मोड़ पर जैसे ही डाकिया बाबू की साइकिल की घंटी ट्रिन-ट्रिन बजती थी, सबके दिल की धड़कनें तेज हो जाती थीं। "
            "वो हल्का नीला इनलैंड लेटर और सफेद पोस्टकार्ड, जिसके आने का इंतजार हफ्तों से रहता था। "
            "जब मां के हाथ में ननिहाल से आई चिट्ठी आती, तो उनकी आंखें खुशी से भर आती थीं। "
            "पूरे परिवार के सामने ऊंची आवाज में एक-एक लफ्ज पढ़कर सुनाया जाता था, जिसमें सबकी खैरियत और गांव की मिट्टी की महक होती थी। "
            "फिर जवाब लिखने के लिए नीली स्याही वाला फाउंटेन पेन और नया अंतर्देशीय पत्र निकाला जाता था। "
            "कोने में 'बड़ों को प्रणाम और छोटों को प्यार' लिखना कोई नहीं भूलता था। "
            "वो शब्द सिर्फ कागज पर लिखी इबारत नहीं थे, बल्कि दिलों को जोड़ने वाला एक सच्चा और पवित्र अहसास थे।"
        ),
        "image_prompts": [
            "A friendly Indian postman in khaki uniform pedaling a black roadster bicycle with a leather mail bag down a leafy suburban lane, ringing the handlebar bell, vertical 9:16",
            "Hands holding a pale blue Indian inland letter aerogramme stamped with a round postmark, delicate handwritten Hindi cursive, vertical 9:16 portrait",
            "An emotional Indian mother sitting on a verandah reading a handwritten letter aloud, tears of joy in her eyes, daughter listening lovingly, vertical 9:16",
            "A fountain pen with blue ink resting on an unfolded postal letter paper beside an inkpot, soft daylight coming through a wooden window, vertical 9:16",
            "A young boy carefully dropping a sealed red letter into a classic cylindrical red street postbox on a quiet corner, vertical 9:16",
            "An elderly couple sitting quietly together holding a stack of tied old letters, smiling fondly at the enduring memories of bygone years, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "inland letter", "indian post", "childhood memories", "handwritten letters", "shorts"],
    },
    {
        "topic": "कैमरे का 36 फोटो वाला रील और फोटो स्टूडियो का रोमांच",
        "youtube_title": "36 फोटो वाला रील और स्टूडियो का वो अनमोल रोमांच",
        "youtube_description": (
            "जब हर फोटो लेने से पहले कहा जाता था: 'कोई हिलेगा नहीं, रील खत्म हो जाएगी!' "
            "36 फोटो के उस एल्बम की सुनहरी यादें। "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "1990 के उस दौर में किसी शादी या जन्मदिन पर जब घर में कैमरे का वो काला डिब्बा खुलता था, तो सबकी सांसें थम जाती थीं। "
            "कोडक का वो 36 फोटो वाला रील, जिसमें हर एक क्लिक की अपनी अनमोल कीमत होती थी। "
            "पापा बड़े ध्यान से कहते थे, कोई हिलेगा नहीं, आंखें खुली रखना, रील खत्म हो जाएगी! "
            "फोटो खींचने के बाद हफ्तों तक स्टूडियो वाले भैया के चक्कर लगाने पड़ते थे कि फोटो धुल कर कब आएगी। "
            "फिर जब वो लिफाफा हाथ में आता, तो पूरे परिवार का मेला लग जाता था। "
            "किसी की आंख बंद हो गई तो सब ठहाका लगाते थे, और अच्छी तस्वीर को प्लास्टिक के एल्बम में बड़े प्यार से सजाया जाता था। "
            "वो 36 तस्वीरों का एल्बम आज के हजारों डिजिटल फोटो से कहीं ज्यादा अनमोल और दिल के करीब था।"
        ),
        "image_prompts": [
            "A vintage 35mm point-and-shoot camera with a yellow Kodak film roll box resting on a polished wooden table, vertical 9:16 portrait",
            "A father peering through the camera viewfinder directing his family standing straight against a flowerpot, shouting instructions with a smile, vertical 9:16",
            "A traditional 1990s Indian photo studio counter with paper photo envelopes and negatives drying in a darkroom background, vertical 9:16",
            "A family eagerly crowding around a dining table opening a newly printed glossy photo packet, laughing and pointing at pictures, vertical 9:16",
            "Carefully sliding a glossy 4x6 family portrait photograph into a clear plastic sleeve of a decorated burgundy photo album, vertical 9:16",
            "Close-up of a classic 1990s retro family photo showing genuine smiles, vintage hairstyles, and cotton clothing, warm nostalgic tint, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "kodak film", "photo album", "family memories", "35mm nostalgia", "shorts"],
    },
    {
        "topic": "गर्मियों की रेल यात्रा, लोहे की खिड़की और कुल्हड़ वाली चाय",
        "youtube_title": "गर्मियों की रेल यात्रा और लोहे की खिड़की वाली वो सीट",
        "youtube_description": (
            "ट्रेन की नीली जालीदार खिड़की वाली सीट, स्टेशन की कुल्हड़ चाय और पूड़ी-अचार का स्वाद। "
            "क्या आपको भी याद है वो नानी के घर जाने का सफर? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "गर्मियों की छुट्टियों में नानी के घर जाने वाली रेलगाड़ी का सफर याद आते ही आज भी दिल खुशियों से झूम उठता है। "
            "लोहे की भारी नीली खिड़की के पास वाली सीट हथियाने की होड़ और ट्रेन की सीटी की वो लंबी गूंज। "
            "जैसे ही गाड़ी रफ्तार पकड़ती, चेहरे पर उड़ती हुई ठंडी हवा और कोयले की भीनी खुशबू का अहसास होता था। "
            "हर छोटे स्टेशन पर कुल्हड़ वाली गर्म चाय और समोसे बेचने वालों की आवाजें गूंजती थीं। "
            "सहयात्रियों के साथ घर से बंधा पूड़ी-आलू और आम का खट्टा अचार बांटकर खाना, अंताक्षरी खेलना और अजनबियों का एक ही सफर में दोस्त बन जाना। "
            "आज राजधानी और शताब्दी में तेजी तो आ गई है, लेकिन लोहे की उस खिड़की से बाहर देखते हुए सुकून भरा वो लंबा सफर आज भी दिल में बसा है।"
        ),
        "image_prompts": [
            "An Indian Railways sleeper coach interior in the 1990s, blue faux-leather berths, a young boy excitedly looking out the iron barred window, vertical 9:16",
            "A steam and diesel train chugging across a scenic rural bridge over a wide river at sunrise, green fields in background, vertical 9:16",
            "A platform chai vendor passing hot steaming tea in terracotta kulhad cups through the train compartment window bars, vertical 9:16 portrait",
            "Family members sharing stainless steel tiffin dabbas of puri, spiced potato curry, and mango pickle on the train lower berth, vertical 9:16",
            "Passengers singing songs and clapping hands together in the aisle as the train sways through the afternoon countryside, vertical 9:16",
            "Distant view of a railway track curving into the twilight horizon with signals glowing red and green, tranquil nostalgia, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "indian railways", "train journey", "kulhad chai", "summer vacation", "shorts"],
    },
    {
        "topic": "गली क्रिकेट, टूटी खिड़की और आखिरी गेंद पर चौका",
        "youtube_title": "गली क्रिकेट, शर्मा जी की खिड़की और आखिरी गेंद का रोमांच",
        "youtube_description": (
            "टेनिस बॉल, तीन ईंटों का विकेट और वो नियम: जिसकी गेंद, उसकी पहली बैटिंग! "
            "बचपन के वो वर्ल्ड कप याद हैं? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "रविवार की दोपहर को जब बड़े-बुजुर्ग खाना खाकर सो जाते थे, तब मोहल्ले की संकरी गली में हमारा वर्ल्ड कप शुरू होता था। "
            "लाल रंग की टेनिस बॉल, लकड़ी का पुराना बैट और तीन ईंटों को जोड़कर बना हुआ वो कामचलाऊ विकेट। "
            "नियम भी बड़े अनोखे थे—जिसकी गेंद, वो पहली बैटिंग करेगा, और गेंद शर्मा जी के घर की खिड़की में गई तो बैट्समैन आउट और गेंद भी वही ढूंढ कर लाएगा। "
            "जब आखिरी गेंद पर जीत के लिए दो रन चाहिए होते थे, तो पूरी गली सांस रोककर बाउंड्री पर खड़ी हो जाती थी। "
            "धूल से सने कपड़े, छिले हुए घुटने और पसीने से तर-बतर चेहरे पर वो जीत की मुस्कान किसी ट्रॉफी से कम नहीं थी। "
            "आज न वो संकरी गलियां रहीं, न वो बेफिक्र दोपहरें, बस यादों का वो मैदान बाकी रह गया है।"
        ),
        "image_prompts": [
            "A narrow Indian residential lane in bright afternoon sun, a young boy batting with an old wooden cricket bat in front of three stacked red bricks, vertical 9:16",
            "A bowler running in with a weathered red tennis ball, neighborhood children fielding alertly along brick walls, vertical 9:16 portrait",
            "The tennis ball breaking through a glass window pane of a ground floor house, children scattering and hiding behind scooters with shocked expressions, vertical 9:16",
            "Tense final ball situation, batsman focusing with sweat dripping down his temple, hands gripping worn bat handle tightly, vertical 9:16",
            "Children jumping and celebrating victory in the dusty lane, waving their bats high in the golden sunset light, vertical 9:16",
            "The quiet evening street after the match, worn bat and ball resting beside the brick wickets under a glowing streetlamp, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "gully cricket", "tennis ball cricket", "indian childhood", "sunday match", "shorts"],
    },
    {
        "topic": "दीपावली पर मिट्टी के दीये, सरसों का तेल और घर की बनी गुझिया",
        "youtube_title": "मिट्टी के दीये, सरसों का तेल और वो पावन दीपावली",
        "youtube_description": (
            "जब दीवाली का मतलब छत पर मिट्टी के दीये धोकर सुखाना और घर की बनी गुझिया खाना था। "
            "बचपन की वो असली रौशनी याद है? "
            "#90sNostalgia #IndianNostalgia #Shorts"
        ),
        "full_narration": (
            "दीपावली आने से कई दिन पहले ही घर की छतों पर मिट्टी के दीयों को पानी की बाल्टी में भिगोकर धूप में सुखाने का नजारा बेहद खूबसूरत होता था। "
            "रूई की लंबी-लंबी बत्तियां हाथों से बटना और दीयों में सरसों का तेल भरकर मुंडेर पर कतार से सजाना। "
            "शाम होते ही माचिस की तीली से एक-एक दीया जलाते हुए हवा से बचाने की वो प्यारी जद्दोजहद। "
            "रसोई से आती शुद्ध घी में छनती गुझिया और बेसन के लड्डुओं की महक पूरे मोहल्ले में फैल जाती थी। "
            "नए कपड़ों की खुशबू और फुलझड़ी घुमाते हुए बच्चों की खिलखिलाहट से पूरी रात रोशन रहती थी। "
            "न कोई दिखावे की महंगी लाइटें थीं और न प्रदूषण का धुआं, बस मिट्टी के दीयों की पवित्र रोशनी और अपनों के साथ से जगमगाती वो सच्ची और पावन दीपावली थी।"
        ),
        "image_prompts": [
            "Dozens of terracotta diya oil lamps soaking in a zinc metal tub of water on an Indian courtyard floor, morning sun reflection, vertical 9:16",
            "An Indian mother and young daughter rolling white cotton wicks between their palms and placing them into clay diyas, vertical 9:16 portrait",
            "A long glowing row of flickering oil diyas along a rooftop terrace parapet against the festive night sky, vertical 9:16",
            "A warm traditional kitchen with a brass kadai frying golden crescent-shaped gujiyas, plates of besan laddoos nearby, vertical 9:16",
            "A young child in traditional festive kurta spinning a bright sparkling golden phuljhadi sparkler with wide joyful eyes, vertical 9:16",
            "Wide panoramic view of a quiet Indian neighborhood glowing solely with hundreds of warm natural terracotta diya flames, vertical 9:16",
        ],
        "tags": ["90s nostalgia", "diwali memories", "mitti ke diye", "traditional diwali", "homemade sweets", "shorts"],
    },
]


def _extract_json(content: str) -> dict[str, Any]:
    """Extract valid JSON from markdown code blocks or raw response string."""
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    elif "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end].strip()

    return json.loads(text)


def _call_groq(preset: ChannelPreset, user_prompt: str, *, temperature: float = 0.75) -> dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable / secret")

    primary_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    candidate_models = [primary_model]
    if primary_model != BACKUP_GROQ_MODEL:
        candidate_models.append(BACKUP_GROQ_MODEL)

    client = Groq(api_key=api_key)
    system_prompt = preset.get("groq_system_hint", "You write engaging YouTube Shorts scripts.")

    last_err: Exception | None = None
    for model_name in candidate_models:
        try:
            log.info("Calling Groq LLM (model=%s, temp=%.2f)...", model_name, temperature)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or ""
            return _extract_json(raw_text)
        except Exception as exc:
            last_err = exc
            log.warning("Groq model %s failed: %s. Trying backup if available...", model_name, exc)

    raise last_err or RuntimeError("All Groq models failed")


def _get_fallback_pack(preset: ChannelPreset, topic: str) -> dict[str, Any]:
    """Rich, validated, novel fallback pack selected from catalog avoiding previous history."""
    from pipeline.story_history import is_duplicate_story, load_history
    import copy
    import random

    history = load_history()
    available: list[dict[str, Any]] = []

    for pack in FALLBACK_CATALOG:
        is_dup, _ = is_duplicate_story(
            topic=pack.get("topic", ""),
            title=pack["youtube_title"],
            narration=pack["full_narration"],
            scene_prompts=pack["image_prompts"],
            history_records=history,
        )
        if not is_dup:
            available.append(pack)

    if available:
        # If user gave a topic hint, see if any available pack matches keywords
        if topic:
            topic_lower = topic.lower()
            matching = [p for p in available if any(w in p["topic"].lower() for w in topic_lower.split() if len(w) > 3)]
            if matching:
                chosen = copy.deepcopy(random.choice(matching))
                chosen["topic"] = topic
                log.info("Selected topic-matched novel fallback pack: '%s'", chosen["youtube_title"])
                return chosen

        chosen = copy.deepcopy(random.choice(available))
        if topic:
            chosen["topic"] = topic
        log.info("Selected novel fallback pack from pool: '%s'", chosen["youtube_title"])
        return chosen

    # If all 10 have been used previously, pick the one least recently used in history
    # (guaranteed not to be the most recent upload)
    log.warning("All catalog fallback packs logged in history; picking least recently used.")
    last_used_index = {}
    for i, h_entry in enumerate(history):
        for pack in FALLBACK_CATALOG:
            if h_entry.get("title") == pack["youtube_title"] or h_entry.get("narration_hash") == pack.get("narration_hash"):
                last_used_index[pack["youtube_title"]] = i

    oldest_pack = min(FALLBACK_CATALOG, key=lambda p: last_used_index.get(p["youtube_title"], -1))
    chosen = copy.deepcopy(oldest_pack)
    if topic:
        chosen["topic"] = topic
    return chosen


def validate_short_pack(
    data: dict[str, Any],
    *,
    min_words: int,
    max_words: int,
    seg_count: int,
) -> tuple[bool, str]:
    """Validate that story pack contains valid schema, narration length, and distinct scene prompts."""
    if not isinstance(data, dict):
        return False, "Response is not a valid JSON dictionary"

    title = str(data.get("youtube_title", "")).strip()
    if not title:
        return False, "Missing or empty youtube_title"

    narration = str(data.get("full_narration", "")).strip()
    word_count = len(narration.split())
    if word_count < min_words:
        return False, f"Narration too short: {word_count} words (minimum required is {min_words})"
    if max_words and word_count > max_words + 30:
        return False, f"Narration too long: {word_count} words (maximum allowed is {max_words})"

    prompts = data.get("image_prompts", [])
    if not isinstance(prompts, list) or len(prompts) != seg_count:
        return False, f"Expected exactly {seg_count} image_prompts, got {len(prompts) if isinstance(prompts, list) else 'invalid'}"

    for i, p in enumerate(prompts):
        if not isinstance(p, str) or len(p.strip()) < 10:
            return False, f"Scene prompt {i + 1} is empty or too short: {p}"

    return True, ""


def generate_short_pack(
    preset: ChannelPreset,
    *,
    topic_hint: str | None = None,
    channel_id: str | None = None,
    avoid_topics: list[str] | None = None,
) -> dict[str, Any]:
    """Generate complete validated Short pack (title, narration, image prompts, tags)."""
    topic = (topic_hint or os.environ.get("SHORT_TOPIC", "")).strip()
    if not topic:
        pool = preset.get("topic_pool", [])
        topic = pool[0] if pool else "1990s Indian childhood nostalgia"

    min_words = int(preset.get("min_words", 130))
    max_words = int(preset.get("max_words", 185))
    seg_count = int(preset.get("segment_count", 6))

    user = (
        f"Channel Preset: {preset['label']}.\n"
        f"Chosen Story Theme / Topic: {topic}.\n"
    )

    if channel_id:
        anti_repeat = history_prompt_block(channel_id, extra_avoid=avoid_topics)
        if anti_repeat:
            user += anti_repeat

    user += f"""
Create ONE complete, emotionally touching YouTube Short script strictly matching the topic.

Return ONLY valid JSON with this exact schema:
{{
  "youtube_title": "concise emotionally compelling title in Hindi or mixed, under 80 characters, no hashtags",
  "youtube_description": "2-3 warm sentences summarizing the story, ending with #90sNostalgia #IndianNostalgia #Shorts",
  "full_narration": "ONE continuous heartfelt narrative paragraph in natural Devanagari Hindi. MUST be between {min_words} and {max_words} words.",
  "image_prompts": [
    "visual description for scene 1: subject, action, retro setting, 1990s props, lighting. In ENGLISH. No text in image.",
    "visual description for scene 2...",
    "..."
  ],
  "tags": ["tag1", "tag2", "tag3"]
}}

MANDATORY RULES:
1. full_narration MUST be {min_words}-{max_words} Hindi words in natural Devanagari script.
2. image_prompts MUST have exactly {seg_count} distinct scene descriptions in English for vertical 9:16.
3. Every scene prompt must be story-specific and visually coherent with the characters and setting.
4. NO text, subtitles, words, logos, or captions inside image prompts.
"""

    max_attempts = 2
    last_err = ""

    # Priority 1 & 2: Groq primary model & backup model
    for attempt in range(max_attempts):
        prompt_with_feedback = user
        if attempt > 0:
            prompt_with_feedback += (
                f"\n\nCRITICAL FIX: Your previous generation failed validation: {last_err}.\n"
                f"Please ensure full_narration has at least {min_words} words and image_prompts has exactly {seg_count} entries.\n"
            )

        try:
            data = _call_groq(preset, prompt_with_feedback, temperature=0.75 if attempt == 0 else 0.4)
            valid, val_err = validate_short_pack(data, min_words=min_words, max_words=max_words, seg_count=seg_count)
            if not valid:
                raise ValueError(val_err)

            data["topic"] = topic
            word_count = len(str(data.get("full_narration", "")).split())
            log.info("Groq script generated successfully (%d words, %d scene prompts)", word_count, len(data.get("image_prompts", [])))
            return data

        except Exception as exc:
            last_err = str(exc)
            log.warning("Groq script generation attempt %d failed: %s", attempt + 1, exc)

    # Priority 3: OpenRouter free-model fallback
    log.warning("All Groq attempts failed. Attempting OpenRouter free fallback...")
    try:
        from pipeline.openrouter import generate_openrouter_short_pack

        openrouter_pack = generate_openrouter_short_pack(
            preset,
            user_prompt=user,
            min_words=min_words,
            max_words=max_words,
            seg_count=seg_count,
            topic=topic,
        )
        if openrouter_pack:
            log.info("OpenRouter free fallback produced validated story pack.")
            return openrouter_pack
    except Exception as openrouter_err:
        log.warning("OpenRouter free fallback encountered unexpected error: %s", openrouter_err)

    # Priority 4: Existing history-filtered 10-story fallback catalog
    log.error("All LLM providers (Groq, OpenRouter) failed. Using validated 90s nostalgia fallback pack.")
    return _get_fallback_pack(preset, topic)
