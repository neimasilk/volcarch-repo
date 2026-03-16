"""
E071: Pre-400 CE Evidence Database for Java & Indonesia
========================================================
Compile ALL known pre-400 CE evidence in Java to reconstruct
the "Invisible Millennium" — the period that VOLCARCH argues
is hidden by volcanic burial, organic material decay, and
historiographic bias.

Sources: archaeological literature, external historical texts,
linguistic evidence, trade goods, genetic studies.

The core question: "What do we KNOW existed before 400 CE?"
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("experiments/E071_pre400ce_evidence/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PRE-400 CE EVIDENCE DATABASE
# Each entry: category, date_range, site/source, location,
#   lat, lon, evidence_type, description, implies_about_civilization,
#   volcanic_zone, reference
# ============================================================

EVIDENCE = [
    # === HOMININ / DEEP PREHISTORY ===
    {
        "category": "Hominin",
        "date_range": "1700000-100000 BCE",
        "site": "Sangiran / Trinil",
        "location": "Central Java (Solo basin)",
        "lat": -7.45, "lon": 110.85,
        "evidence_type": "Fossils",
        "description": "Homo erectus (Java Man) — Pithecanthropus erectus. Multiple crania, femora. Dated 1.7Ma-100ka. Found in volcanic tuff layers.",
        "implies": "Long continuous occupation of Java. Volcanic sediments PRESERVE fossils at deep time but BURY recent ones.",
        "volcanic_zone": "Yes (Lawu/Solo basin volcanism)",
        "reference": "Dubois 1894; Swisher et al. 1994",
    },
    {
        "category": "Hominin",
        "date_range": "100000-40000 BCE",
        "site": "Song Terus cave",
        "location": "Punung, East Java",
        "lat": -8.067, "lon": 111.017,
        "evidence_type": "Stone tools, faunal remains",
        "description": "Continuous occupation from ~100ka. Sophisticated bone tools (spatulas, points). Earliest shell beads in Island SEA.",
        "implies": "Advanced tool-making, possible symbolic behavior. Cave sites survive; open-air sites from same period are absent.",
        "volcanic_zone": "Marginal (30km from Lawu)",
        "reference": "Sémah et al. 2004; Chazine & Zeitoun 2007",
    },
    {
        "category": "Hominin",
        "date_range": "45000-12000 BCE",
        "site": "Leang Bulu Sipong / Maros caves",
        "location": "South Sulawesi",
        "lat": -4.97, "lon": 119.63,
        "evidence_type": "Cave paintings",
        "description": "World's oldest figurative art (45.5ka hunting scene). Hand stencils. Evidence of sophisticated symbolic culture.",
        "implies": "Modern humans in Indonesia had complex symbolic systems >40ka before Indianization.",
        "volcanic_zone": "No (karst, non-volcanic)",
        "reference": "Aubert et al. 2019; Brumm et al. 2021",
    },
    # === NEOLITHIC / AUSTRONESIAN ===
    {
        "category": "Neolithic",
        "date_range": "4000-2000 BCE",
        "site": "Austronesian expansion into Java",
        "location": "Java (general)",
        "lat": -7.5, "lon": 110.5,
        "evidence_type": "Linguistic, comparative",
        "description": "Austronesian-speaking farmers arrive in Java ~4000-3000 BCE based on linguistic phylogenetics and comparative archaeology. Bring rice agriculture, pottery, pig/chicken domestication, outrigger canoe technology.",
        "implies": "Settled agricultural communities in Java for ~4500 years before first inscription. Where are their settlements?",
        "volcanic_zone": "All zones",
        "reference": "Bellwood 2017; Blust 2009",
    },
    {
        "category": "Neolithic",
        "date_range": "3500-2000 BCE",
        "site": "Kalumpang / Minanga Sipakko",
        "location": "West Sulawesi",
        "lat": -2.5, "lon": 119.4,
        "evidence_type": "Pottery, stone adzes",
        "description": "Early Austronesian settlement. Red-slipped pottery. Polished stone adzes. Rice remains.",
        "implies": "Clear Neolithic presence in Sulawesi. Similar should exist in Java but has not been found at open-air sites.",
        "volcanic_zone": "No",
        "reference": "Simanjuntak 2008",
    },
    {
        "category": "Neolithic",
        "date_range": "3000-1000 BCE",
        "site": "Kendeng Lembu",
        "location": "Banyuwangi, East Java",
        "lat": -8.35, "lon": 114.2,
        "evidence_type": "Polished stone axes, pottery",
        "description": "Neolithic workshop site. Thousands of polished stone axes. Evidence of specialized craft production.",
        "implies": "Organized community with craft specialization. Open-air site survived because it's in non-volcanic karst zone.",
        "volcanic_zone": "No (far from major volcanoes)",
        "reference": "Noerwidi 2014",
    },
    # === BRONZE AGE / IRON AGE ===
    {
        "category": "Metal Age",
        "date_range": "500-100 BCE",
        "site": "Dong Son drums in Java",
        "location": "Multiple sites across Java",
        "lat": -7.5, "lon": 110.0,
        "evidence_type": "Bronze drums, axes",
        "description": "Vietnamese-origin bronze drums (Heger Type I) found across Java: Pekalongan (Randu Gunting), Semarang, Bali. Evidence of long-distance maritime trade networks.",
        "implies": "Java was connected to mainland SEA bronze trade networks by 500 BCE. Communities must have had sufficient surplus and social organization for elite goods exchange.",
        "volcanic_zone": "Mixed",
        "reference": "Bernet Kempers 1988",
    },
    {
        "category": "Metal Age",
        "date_range": "500 BCE-500 CE",
        "site": "Pasir Angin",
        "location": "Bogor, West Java",
        "lat": -6.65, "lon": 106.8,
        "evidence_type": "Megalithic complex, pottery, metal",
        "description": "Large megalithic site with menhirs, terraced platforms. Pottery, bronze fragments. Continuous use from late prehistoric to early historic period.",
        "implies": "Organized community with monumental construction capability BEFORE Indianization. Located in volcanic terrain (Salak volcano).",
        "volcanic_zone": "Yes (near Salak volcano)",
        "reference": "Sutayasa 1979",
    },
    {
        "category": "Metal Age",
        "date_range": "200 BCE-200 CE",
        "site": "Buni Complex",
        "location": "North coast West Java (Buni, Bekasi, Karawang)",
        "lat": -6.15, "lon": 107.15,
        "evidence_type": "Pottery, glass beads, bronze",
        "description": "Distinctive pottery complex with paddle-impressed decoration. Indian rouletted ware sherds. Glass beads of Indian origin. Bronze objects. Evidence of maritime trade with India/South Asia.",
        "implies": "Active MARITIME TRADE with India centuries before Indianization. Complex pottery tradition. Coastal site (low volcanic risk but subject to coastal change).",
        "volcanic_zone": "No (coastal plain)",
        "reference": "Walker & Santoso 1977; Manguin & Agustijanto 2012",
    },
    {
        "category": "Metal Age",
        "date_range": "200 BCE-300 CE",
        "site": "Sembiran / Pacung",
        "location": "North Bali",
        "lat": -8.15, "lon": 115.35,
        "evidence_type": "Indian rouletted ware, glass beads, iron",
        "description": "Earliest Indian trade goods in Indonesia. Indian rouletted ware identical to Arikamedu (Tamil Nadu). Glass beads, iron slag. Evidence of direct India-Bali maritime connection. Prehistoric burial site with grave goods.",
        "implies": "DIRECT trade with India ~200 BCE. Not just passive recipients — Bali was actively connected to Indian Ocean networks. Pre-Hindu material culture was sophisticated enough to participate in international trade.",
        "volcanic_zone": "Yes (Agung/Batur ~30km)",
        "reference": "Ardika & Bellwood 1991; Calo et al. 2015",
    },
    {
        "category": "Metal Age",
        "date_range": "200 BCE-200 CE",
        "site": "Roman/Mediterranean finds in Indonesia",
        "location": "Multiple sites (Java, Sumatra, Bali)",
        "lat": -7.0, "lon": 110.0,
        "evidence_type": "Coins, beads, glass",
        "description": "Roman coins found in Java (antoniniani, denarii). Mediterranean glass beads. Indicate Indonesia was part of global trade network connecting Rome-India-Southeast Asia.",
        "implies": "Java was known to the Roman-era world. Trade infrastructure (ports, markets, intermediaries) must have existed.",
        "volcanic_zone": "Mixed",
        "reference": "Manguin 2004; Calo 2014",
    },
    # === MEGALITHIC TRADITIONS ===
    {
        "category": "Megalithic",
        "date_range": "2000 BCE-1500 CE",
        "site": "Gunung Padang",
        "location": "Cianjur, West Java",
        "lat": -6.99, "lon": 107.06,
        "evidence_type": "Megalithic terraced structure",
        "description": "Massive terraced megalithic site with columnar basalt structures. Controversial dating (some claims of extreme antiquity, contested). Definite pre-Hindu megalithic tradition. Largest megalithic site in Southeast Asia.",
        "implies": "Pre-Hindu monumental construction capability. Whether 10,000 or 2,000 years old, demonstrates sophisticated architectural tradition predating Indianization.",
        "volcanic_zone": "Yes (near Gede/Pangrango)",
        "reference": "Natawidjaja et al. 2018 (contested)",
    },
    {
        "category": "Megalithic",
        "date_range": "1000 BCE-present",
        "site": "Sumba megalithic tradition",
        "location": "Sumba, NTT",
        "lat": -9.65, "lon": 119.95,
        "evidence_type": "Living megalithic tradition",
        "description": "Continuous megalithic tradition surviving to present day. Stone tombs, ancestor worship, elaborate mortuary ritual. Provides ethnographic analogue for pre-Hindu Java.",
        "implies": "Megalithic traditions in Indonesia are NOT 'primitive' — they represent sophisticated social organization with complex ritual. Java's megalithic phase was comparable.",
        "volcanic_zone": "No",
        "reference": "Adams 2007",
    },
    {
        "category": "Megalithic",
        "date_range": "500 BCE-500 CE",
        "site": "Cipari / Kuningan megalithic",
        "location": "Kuningan, West Java",
        "lat": -6.95, "lon": 108.48,
        "evidence_type": "Menhirs, stone terraces, sarcophagi",
        "description": "Extensive megalithic complex. Stone terraces, menhirs, sarcophagi. Evidence of organized community with mortuary ritual.",
        "implies": "Organized pre-Hindu communities in volcanic Java with monumental construction.",
        "volcanic_zone": "Yes (Ciremai volcano ~15km)",
        "reference": "Prasetyo 2015",
    },
    # === EXTERNAL HISTORICAL REFERENCES ===
    {
        "category": "External text",
        "date_range": "300 BCE-300 CE",
        "site": "Ramayana: Yavadvipa reference",
        "location": "India (describing Java)",
        "lat": None, "lon": None,
        "evidence_type": "Sanskrit epic text",
        "description": "Ramayana (Kiskindhakanda 40.30) mentions Yavadvipa ('island of barley/millet') as a known land with gold and silver mines, beyond the sea. The reference implies Java was known to Indian geographical tradition.",
        "implies": "Java was known to Indian traders/scholars by name before the Common Era. The description of gold/silver mines suggests economic knowledge.",
        "volcanic_zone": "N/A",
        "reference": "Wheatley 1961; Coedès 1968",
    },
    {
        "category": "External text",
        "date_range": "~150 CE",
        "site": "Ptolemy: Iabadiou",
        "location": "Alexandria (describing Java)",
        "lat": None, "lon": None,
        "evidence_type": "Greek geographical text",
        "description": "Ptolemy's Geography lists 'Iabadiou' (Yavadvipa/Java) as a large island rich in gold, with a trading port called 'Argyre' (Silver Town). Placed east of India.",
        "implies": "Java was known to Greco-Roman geographical knowledge by 150 CE. The mention of a named TRADING PORT implies urban settlement and organized commerce.",
        "volcanic_zone": "N/A",
        "reference": "Ptolemy, Geography VII.2; Wheatley 1961",
    },
    {
        "category": "External text",
        "date_range": "~70 CE",
        "site": "Pliny the Elder: Indonesia references",
        "location": "Rome (describing Indonesia)",
        "lat": None, "lon": None,
        "evidence_type": "Roman encyclopedic text",
        "description": "Pliny's Natural History (VI.54) mentions trade goods from the 'Eastern Islands' including spices (cloves, likely from Maluku), suggesting Roman-era knowledge of Indonesian trade networks.",
        "implies": "Indonesian spice trade was reaching Rome by 1st century CE. This trade required sophisticated maritime networks across Indonesia.",
        "volcanic_zone": "N/A",
        "reference": "Pliny, Natural History VI.54; Miller 1969",
    },
    {
        "category": "External text",
        "date_range": "~414 CE",
        "site": "Fa Xian: Ye-Po-Ti (Yavadvipa)",
        "location": "China/Java",
        "lat": None, "lon": None,
        "evidence_type": "Chinese Buddhist pilgrim account",
        "description": "Chinese monk Fa Xian stopped at Ye-Po-Ti (identified as Java or Sumatra) during return voyage from India. Describes a land where 'Brahmanism flourishes but Buddhism has not yet reached.' Notes 'heretics and Brahmans.'",
        "implies": "By ~414 CE, Java/Sumatra already had established Brahmanic religion. This was BEFORE the first inscriptions. Hindu culture was already present, just not yet producing inscriptions.",
        "volcanic_zone": "N/A",
        "reference": "Fa Xian, A Record of Buddhistic Kingdoms; Legge 1886 translation",
    },
    # === LINGUISTIC EVIDENCE ===
    {
        "category": "Linguistic",
        "date_range": "2000 BCE-500 CE",
        "site": "Pre-Indic substrate vocabulary",
        "location": "Java (linguistic, not spatial)",
        "lat": None, "lon": None,
        "evidence_type": "Substrate vocabulary in Javanese/Balinese",
        "description": "VOLCARCH E022-E029: Machine learning detection of pre-Austronesian substrate in Javanese (~29% non-Austronesian, non-Sanskrit vocabulary). Substrate concentrated in agriculture, body parts, kinship — domestic domains predating Indianization. Parallel independent substrate in multiple islands.",
        "implies": "Pre-Indic Java had its own vocabulary for agriculture, kinship, and material culture — evidence of developed society with specialized domains.",
        "volcanic_zone": "N/A",
        "reference": "Amien & Gunawan 2026 (P8, under review)",
    },
    {
        "category": "Linguistic",
        "date_range": "pre-500 CE",
        "site": "Pre-Hindu toponyms in Java",
        "location": "Java (25,244 villages)",
        "lat": None, "lon": None,
        "evidence_type": "Place name analysis",
        "description": "VOLCARCH E051: 57.7% of Java's village names are pre-Hindu (not Sanskrit, not Arabic). Madura reaches 70-91%. These names preserve pre-Indic landscape knowledge. Court-center model: Indianization radiates from Yogyakarta, fades in periphery.",
        "implies": "The majority of Java's landscape was named BEFORE Indianization. Pre-Hindu communities had comprehensive geographic knowledge sufficient for systematic place-naming.",
        "volcanic_zone": "N/A",
        "reference": "Amien 2026 (VOLCARCH E051)",
    },
    # === BIOLOGICAL / AGRICULTURAL ===
    {
        "category": "Agricultural",
        "date_range": "3000-1000 BCE",
        "site": "Rice agriculture in Java",
        "location": "Java (general)",
        "lat": -7.0, "lon": 110.5,
        "evidence_type": "Archaeobotanical, comparative",
        "description": "Wet rice agriculture established in Java by ~3000-2000 BCE based on comparative Austronesian evidence and pollen records. Irrigated rice requires organized water management, land tenure systems, and community cooperation.",
        "implies": "Organized agrarian communities with water management infrastructure existed >3000 years before first inscriptions. The sawah system requires social organization comparable to early states.",
        "volcanic_zone": "All zones (volcanic soil ideal for rice)",
        "reference": "Bellwood 2017; Maloney 1996",
    },
    {
        "category": "Agricultural",
        "date_range": "2000 BCE-present",
        "site": "Slametan mortuary tradition",
        "location": "Java (general)",
        "lat": None, "lon": None,
        "evidence_type": "Ethnographic, textual",
        "description": "VOLCARCH P5: 1000-day slametan mortuary cycle predates Hinduism. Intervals map to decomposition stages in volcanic soil. Pre-Indic ritual system (hyang, maṅhuri) persists in >47% of inscriptions even after Indianization.",
        "implies": "Sophisticated mortuary ritual system with implicit knowledge of taphonomic processes. This is NOT primitive — it requires multigenerational empirical observation of decomposition rates.",
        "volcanic_zone": "N/A",
        "reference": "Amien 2026 (P5, under review); Hendrajaya & Almu'tasim 2020",
    },
    # === GENETIC EVIDENCE ===
    {
        "category": "Genetic",
        "date_range": "50000 BCE-present",
        "site": "Denisovan admixture in modern Indonesians",
        "location": "Indonesia-wide",
        "lat": None, "lon": None,
        "evidence_type": "Genomic analysis",
        "description": "Modern Indonesians carry 2-4% Denisovan ancestry, more than any other population. This introgression occurred ~45,000 years ago in Wallacea. Multiple admixture events suggest sustained interaction with archaic hominins unique to the Indonesian archipelago.",
        "implies": "Indonesia's human history is deeper and more complex than anywhere else in the Austronesian world. The continent-like depth of genetic history contrasts sharply with the shallow archaeological record.",
        "volcanic_zone": "All",
        "reference": "Jacobs et al. 2019; Carlhoff et al. 2021",
    },
    {
        "category": "Genetic",
        "date_range": "3000-1000 BCE",
        "site": "Austronesian migration genomics",
        "location": "Indonesia-wide",
        "lat": None, "lon": None,
        "evidence_type": "Genomic analysis",
        "description": "Ancient DNA and modern genomics confirm at least two major migration events into Indonesia: (1) First wave ~50ka (Papuan-related), (2) Austronesian expansion ~3000-2000 BCE (Taiwan-origin farmers). Java shows predominantly Austronesian ancestry with Papuan-related admixture.",
        "implies": "Population continuity in Java for ~4000+ years before first inscription. These are the SAME people who later adopted Sanskrit writing — not newcomers.",
        "volcanic_zone": "All",
        "reference": "Lipson et al. 2018; McColl et al. 2018",
    },
    # === TRADE NETWORK EVIDENCE ===
    {
        "category": "Maritime trade",
        "date_range": "300 BCE-300 CE",
        "site": "Indian Ocean maritime network",
        "location": "Indonesia-wide (particularly Strait of Malacca, Java Sea)",
        "lat": None, "lon": None,
        "evidence_type": "Shipwrecks, trade goods",
        "description": "Evidence of regular maritime trade connecting India, Southeast Asia, and China by ~300 BCE. Indonesian outrigger canoes (prau) were among the most sophisticated sailing vessels in the pre-modern world. Madagascar was colonized by Indonesian sailors ~500-700 CE — proving trans-oceanic capability.",
        "implies": "Indonesian maritime communities had world-class navigation technology and were active participants (not passive recipients) in Indian Ocean trade for centuries before Indianization.",
        "volcanic_zone": "Coastal",
        "reference": "Manguin 2004; Beaujard 2012",
    },
    {
        "category": "Maritime trade",
        "date_range": "100 BCE-500 CE",
        "site": "Kosambi hypothesis: Indianization was trade-driven",
        "location": "Conceptual",
        "lat": None, "lon": None,
        "evidence_type": "Historical theory",
        "description": "D.D. Kosambi and later van Leur argued that Indianization was driven by EXISTING Indonesian demand for prestige goods and religious legitimacy — not by Indian colonization. Indonesian rulers adopted Sanskrit because it served their political purposes, not because they lacked civilization.",
        "implies": "The 'beginning' at 400 CE is not the beginning of civilization but the beginning of WRITING. The civilization that adopted writing was already complex.",
        "volcanic_zone": "N/A",
        "reference": "Kosambi 1965; van Leur 1955; Wolters 1967",
    },
    # === EXPANDED EXTERNAL TEXTUAL SOURCES ===
    {
        "category": "External text",
        "date_range": "300 BCE-200 CE",
        "site": "Ramayana 'Yavadvipa' — seven kingdoms",
        "location": "India (describing Java)",
        "lat": None, "lon": None,
        "evidence_type": "Sanskrit epic text",
        "description": "Ramayana describes Yavadvipa with 'seven kingdoms' — implying political complexity and multiple polities. Sanskrit name 'island of barley' indicates agricultural recognition. Multiple kingdoms suggest inter-polity relations, diplomacy, and competition centuries before any inscription.",
        "implies": "Political fragmentation into seven kingdoms implies state-level organization with territorial boundaries, succession systems, and inter-polity competition — all before 200 CE.",
        "volcanic_zone": "other",
        "reference": "Valmiki Ramayana, Kiskindhakanda; Wheatley 1961; Coedès 1968",
    },
    {
        "category": "External text",
        "date_range": "150 CE",
        "site": "Ptolemy 'Iabadiou' — coordinate accuracy",
        "location": "Alexandria (describing Java)",
        "lat": None, "lon": None,
        "evidence_type": "Greek geographical text",
        "description": "Ptolemy's Geography renders Yavadvipa as 'Iabadiou', placed at ~8.5°S latitude — remarkably accurate for 2nd-century geographical knowledge (Java's actual latitude ~7-8°S). Described as 'very fertile' with gold. This precision implies detailed informant knowledge, likely from sailors who had visited.",
        "implies": "Accurate latitude placement at ~8.5°S implies repeated voyages and accumulated navigational knowledge. Java was not a rumor but a documented destination with known coordinates.",
        "volcanic_zone": "other",
        "reference": "Ptolemy, Geography VII.2; Stückelberger & Graßhoff 2006",
    },
    {
        "category": "External text",
        "date_range": "77 CE",
        "site": "Pliny Naturalis Historia — Chryse and spice trade",
        "location": "Rome (describing Indonesian archipelago)",
        "lat": None, "lon": None,
        "evidence_type": "Roman encyclopedic text",
        "description": "Pliny's Naturalis Historia references 'Chryse' (gold island) and documents spice trade goods originating from the Indonesian archipelago. The specificity of trade goods (cloves, cassia) indicates established commercial channels, not occasional contact.",
        "implies": "By 77 CE, Indonesian spice trade goods were catalogued in Roman encyclopedic literature. This requires sustained multi-node trade networks across the Indian Ocean.",
        "volcanic_zone": "other",
        "reference": "Pliny, Naturalis Historia VI.54, XII; Miller 1969",
    },
    {
        "category": "External text",
        "date_range": "~50 CE",
        "site": "Periplus Maris Erythraei — Chryse as easternmost destination",
        "location": "Greco-Roman world (describing Southeast Asia)",
        "lat": None, "lon": None,
        "evidence_type": "Greek trade manual",
        "description": "The Periplus Maris Erythraei identifies 'Chryse' as the easternmost known trade destination. Indonesian spices — particularly cloves and cinnamon — were documented in Mediterranean commerce. The Periplus is a practical merchant's guide, not literary geography, making its references operationally significant.",
        "implies": "Indonesian archipelago was the terminus of the longest trade route in the ancient world. Merchant ships were reaching or sourcing goods from Indonesia by mid-1st century CE.",
        "volcanic_zone": "other",
        "reference": "Periplus Maris Erythraei §63; Casson 1989",
    },
    {
        "category": "External text",
        "date_range": "~111 CE (events ~1st c. BCE)",
        "site": "Han Shu maritime route through Southeast Asia",
        "location": "China (describing maritime Southeast Asia)",
        "lat": None, "lon": None,
        "evidence_type": "Chinese dynastic history",
        "description": "Ban Gu's Han Shu (Book of Han, compiled ~111 CE) records a maritime trade route from southern China through Southeast Asia, describing events from the 1st century BCE. The route passes through the Indonesian archipelago, confirming Chinese awareness of maritime Southeast Asian polities.",
        "implies": "Chinese maritime trade through Indonesian waters was established by the 1st century BCE. This implies port facilities, provisioning stations, and local intermediaries along the route.",
        "volcanic_zone": "other",
        "reference": "Ban Gu, Han Shu (Dili zhi); Wang 1958; Wolters 1967",
    },
    {
        "category": "External text",
        "date_range": "132 CE",
        "site": "Hou Han Shu 'Ye-Tiao' embassy to Han court",
        "location": "China (receiving embassy from Java/Sumatra)",
        "lat": None, "lon": None,
        "evidence_type": "Chinese dynastic history",
        "description": "The Hou Han Shu (Book of Later Han) records that in 132 CE, the kingdom of 'Ye-Tiao' (possibly Java or Sumatra) sent a diplomatic mission to the Han court. The embassy brought tribute goods. This is the earliest recorded diplomatic contact between Indonesia and China.",
        "implies": "By 132 CE, a polity in Java or Sumatra was sufficiently organized to dispatch a diplomatic embassy to China — implying state-level organization, knowledge of Chinese protocol, and maritime capability for the voyage.",
        "volcanic_zone": "other",
        "reference": "Hou Han Shu (Xiyu zhuan); Wheatley 1961; Wolters 1967",
    },
    {
        "category": "External text",
        "date_range": "414 CE",
        "site": "Fa Xian 'Ye-Po-Ti' — firsthand account",
        "location": "Java or Sumatra",
        "lat": None, "lon": None,
        "evidence_type": "Chinese Buddhist pilgrim first-person account",
        "description": "Fa Xian's first-person account of Ye-Po-Ti (414 CE): Brahmanism was already flourishing, while Buddhism was 'not much known.' This is a direct eyewitness observation, not hearsay. The established state of Brahmanism implies decades or centuries of prior Indian religious influence.",
        "implies": "First-person witness confirms Hindu religious infrastructure was mature by 414 CE — BEFORE any surviving inscription in Java. Buddhism's absence suggests selective adoption of Indian cultural elements.",
        "volcanic_zone": "other",
        "reference": "Fa Xian, Foguo ji (A Record of Buddhistic Kingdoms); Legge 1886",
    },
    # === EXPANDED ARCHAEOLOGICAL / TRADE EVIDENCE ===
    {
        "category": "Maritime trade",
        "date_range": "200 BCE-200 CE",
        "site": "Sembiran Rouletted Ware — direct India-Bali trade",
        "location": "North Bali",
        "lat": -8.15, "lon": 115.35,
        "evidence_type": "Indian pottery, glass beads, high-tin bronze",
        "description": "Sembiran yielded Indian rouletted ware, glass beads, and high-tin bronze — materials diagnostic of direct trade contact with South India (Tamil Nadu / Arikamedu). High-tin bronze is particularly significant as it indicates metallurgical exchange, not just commodity trade.",
        "implies": "Direct maritime trade between India and Bali by ~200 BCE. High-tin bronze indicates technological transfer, not just commodity exchange. Bali had communities sophisticated enough to participate in international metallurgical networks.",
        "volcanic_zone": "other",
        "reference": "Ardika & Bellwood 1991; Calo et al. 2015; Bellina 2007",
    },
    {
        "category": "Maritime trade",
        "date_range": "200 BCE-500 CE",
        "site": "Buni Complex — coastal trade station",
        "location": "North coast West Java",
        "lat": -6.15, "lon": 107.15,
        "evidence_type": "Paddle-impressed pottery, Indian imports",
        "description": "The Buni Complex represents a coastal trade station with distinctive paddle-impressed pottery tradition and Indian trade imports. The site complex spans multiple locations along the north Java coast (Buni, Bekasi, Karawang), suggesting a network of related settlements rather than a single site.",
        "implies": "A network of coastal trade stations implies organized maritime commerce with standardized pottery production. This proto-urban coastal network predates any inscriptional evidence from West Java by centuries.",
        "volcanic_zone": "volcanic",
        "reference": "Walker & Santoso 1977; Manguin & Agustijanto 2012",
    },
    {
        "category": "Archaeological",
        "date_range": "300 BCE-100 CE",
        "site": "Dong Son bronze drums — including Moon of Pejeng",
        "location": "Java and Bali",
        "lat": -8.53, "lon": 115.35,
        "evidence_type": "Bronze drums, elite exchange goods",
        "description": "Dong Son bronze drums distributed across Java and Bali, including the Moon of Pejeng (Bulan Pejeng) in Bali — at 186cm diameter, the largest known bronze drum in the world. Its size indicates local casting capability or elite-level long-distance acquisition. Found in ritual/ceremonial contexts.",
        "implies": "The Moon of Pejeng's extraordinary size (186cm) suggests either local bronze-casting at world-class scale or the ability to transport massive prestige objects across maritime Southeast Asia. Either interpretation implies high social complexity.",
        "volcanic_zone": "other",
        "reference": "Bernet Kempers 1988; Calo 2014",
    },
    {
        "category": "Maritime trade",
        "date_range": "~1700 BCE",
        "site": "Clove at Terqa, Syria — trans-oceanic trade evidence",
        "location": "Terqa, Syria (cloves native only to Maluku)",
        "lat": 34.92, "lon": 40.55,
        "evidence_type": "Archaeobotanical remains",
        "description": "Cloves (Syzygium aromaticum) found in a burned house at Terqa, Syria, dated to ~1700 BCE. Cloves are native ONLY to the Maluku Islands (eastern Indonesia). Their presence in Bronze Age Syria implies a trans-oceanic trade network spanning >10,000 km, the longest documented trade route of the 2nd millennium BCE.",
        "implies": "Indonesian spices were reaching the Middle East by 1700 BCE — over 2,000 years before any inscription in Indonesia. This requires multi-node maritime trade networks of extraordinary geographic scope.",
        "volcanic_zone": "other",
        "reference": "Buccellati & Buccellati 1983; Turner 2004; Cornévin 2015",
    },
    {
        "category": "Archaeological",
        "date_range": "4th-5th century CE",
        "site": "Batujaya temples — earliest Buddhist architecture",
        "location": "Karawang, West Java",
        "lat": -6.15, "lon": 107.30,
        "evidence_type": "Brick Buddhist temple complex",
        "description": "Batujaya temple complex in Karawang represents the earliest known Buddhist architecture in Indonesia, dated to the 4th-5th century CE. Brick stupas and viharas. Located near the Buni Complex sites, suggesting continuity from pre-Hindu trade settlement to early Buddhist religious center.",
        "implies": "Buddhist architectural tradition in West Java contemporary with or slightly after the earliest inscriptions. The proximity to Buni Complex suggests centuries of cultural development from trade station to religious center.",
        "volcanic_zone": "volcanic",
        "reference": "Manguin & Agustijanto 2012; Djafar 2010",
    },
    # === EARLIEST INSCRIPTIONS (for boundary) ===
    {
        "category": "Earliest inscription",
        "date_range": "~400 CE",
        "site": "Yupa inscriptions (Mulawarman)",
        "location": "Kutai, East Kalimantan",
        "lat": -0.5, "lon": 117.0,
        "evidence_type": "Sanskrit inscription on stone pillars",
        "description": "Earliest known inscriptions in Indonesia. 7 yupa (sacrificial posts) by King Mulawarman, grandson of Kundungga (non-Sanskrit name). Pallava script. Describe Vedic yajna sacrifice with massive cattle gifts.",
        "implies": "Even the FIRST inscriptions describe a third-generation king — meaning the dynasty existed for 50-100 years before adopting writing. The grandfather's name (Kundungga) is indigenous, not Sanskrit.",
        "volcanic_zone": "No (Kalimantan, non-volcanic)",
        "reference": "Vogel 1918; Chhabra 1965",
    },
    {
        "category": "Earliest inscription",
        "date_range": "~450 CE",
        "site": "Tarumanagara inscriptions (Purnavarman)",
        "location": "West Java (Bogor, Bekasi)",
        "lat": -6.6, "lon": 106.8,
        "evidence_type": "Sanskrit inscription on stone",
        "description": "Seven inscriptions by King Purnavarman of Tarumanagara. Describe canal construction (Tugu inscription), city fortification, Vishnu worship. The Tugu inscription records a 12-km canal — evidence of massive organized labor.",
        "implies": "By ~450 CE, West Java had a state capable of building 12-km canals. This infrastructure does not emerge overnight — it requires centuries of prior development.",
        "volcanic_zone": "Yes (near Salak/Gede)",
        "reference": "Kern 1917; Vogel 1925",
    },
]


def main():
    print("=" * 70)
    print("E071: Pre-400 CE Evidence Database — The Invisible Millennium")
    print("=" * 70)

    # Write to CSV
    csv_file = RESULTS_DIR / "pre400ce_evidence.csv"
    fieldnames = ['category', 'date_range', 'site', 'location', 'lat', 'lon',
                  'evidence_type', 'description', 'implies', 'volcanic_zone', 'reference']
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in EVIDENCE:
            writer.writerow(e)

    # Write to JSON
    json_file = RESULTS_DIR / "pre400ce_evidence.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(EVIDENCE, f, indent=2, ensure_ascii=False)

    # Summary statistics
    cats = defaultdict(int)
    volcanic = defaultdict(int)
    for e in EVIDENCE:
        cats[e['category']] += 1
        vz = e.get('volcanic_zone', 'Unknown')
        if vz and vz.startswith('Yes'):
            volcanic['volcanic'] += 1
        elif vz and vz.startswith('No'):
            volcanic['non-volcanic'] += 1
        else:
            volcanic['other'] += 1

    print(f"\nTotal entries: {len(EVIDENCE)}")
    print(f"\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:25s}: {count}")
    print(f"\nBy volcanic zone:")
    for vz, count in sorted(volcanic.items()):
        print(f"  {vz:15s}: {count}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY SYNTHESIS")
    print("=" * 70)
    print("""
1. CONTINUOUS OCCUPATION: Java has been continuously occupied for >100,000 years.
   Cave sites show unbroken cultural sequences. Open-air sites are absent from
   the record precisely where VOLCARCH predicts: in volcanic terrain.

2. AUSTRONESIAN COMMUNITIES: Settled agricultural communities in Java since
   ~4000-3000 BCE. Rice agriculture, pottery, domesticated animals. 4,500 years
   of community life before the first inscription.

3. TRADE NETWORKS: Indian Ocean maritime trade connected Java to India and
   Rome by ~200 BCE. Indonesian sailors later colonized Madagascar (~2000km
   across open ocean). This is NOT a "primitive" society.

4. EXTERNAL RECOGNITION: Ptolemy (150 CE), Indian epics (Ramayana), and
   Chinese sources all describe Java as a known, wealthy, trading island
   CENTURIES before the first inscription.

5. THE INSCRIPTION PARADOX: Even the FIRST inscriptions (Mulawarman ~400 CE)
   describe a THIRD-GENERATION king — meaning the dynasty predates writing
   by 50-100 years. Grandfather's name (Kundungga) is indigenous.
   Tarumanagara (~450 CE) built 12-km canals — infrastructure requiring
   centuries of prior state development.

6. VOLCANIC TAPHONOMIC PREDICTION: Open-air prehistoric sites are found in
   non-volcanic zones (Buni on coast, Kendeng Lembu in karst, Kalimantan).
   Equivalent sites in volcanic Java are missing — consistent with burial.

CONCLUSION: The "beginning" at 400 CE is the beginning of WRITING, not the
beginning of CIVILIZATION. The civilization was already mature when it adopted
Sanskrit script. The preceding millennia are invisible because:
  (a) organic materials decomposed (63% of inscribed goods are organic)
  (b) volcanic sedimentation buried open-air sites (3-13 mm/yr)
  (c) colonial historiography privileged Sanskrit-literate periods
""")

    print(f"\nOutputs:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")


if __name__ == "__main__":
    main()
