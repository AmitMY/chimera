import pickle
from itertools import chain

from utils.dbpedia import gender_pronouns, all_pronouns

reg = pickle.load(
    open("cache/WebNLG_Exp/model-feats/translate-neural/translate-verify/eval-bert-reg/ents-reg-map.sav", "rb"))

print(reg)


print("Total cases", sum(reg.values()))
print("Total cases where the replacement is self", sum([v for (k1, k2), v in reg.items() if k1 == k2]))
print("Total cases where the replacement is self by BERT",
      sum([v for (k1, k2), v in reg.items() if k1 != k2 and k1.lower() == k2.lower()]))

print("Total cases where the replacement is self with 'the'",
      sum([v for (k1, k2), v in reg.items() if "the " + k1 == k2]))
print("Total cases where the replacement is a pronoun", sum([v for (k1, k2), v in reg.items() if k2 in all_pronouns]))

print()

stuff = """ADARE MANOR	:	manor	:	3
ALL INDIA COUNCIL FOR TECHNICAL EDUCATION:the it: 2
ACCADEMIA DI ARCHITETTURA DI MENDRISIO:the it: 3
JOHN MADIN:john:2
ALAN BEAN	:	alan	:	9
ALAN BEAN	:	bean	:	3
ALAN SHEPARD	:	alan	:	19
ALAN SHEPARD	:	shepard	:	1
BUZZ ALDRIN	:	buzz	:	36
ERNIE COLÓN	:	ernie	:	3
JENS HÄRTEL	:	jens	:	1
JOHN VAN DEN BROM	:	john	:	3
MASSIMO DRAGO	:	massimo	:	3
PETER STÖGER	:	peter	:	2
WILLIAM ANDERS	:	william	:	11
WILLIAM ANDERS	:	anders	:	3
ADAMS COUNTY, PENNSYLVANIA	:	pennsylvania	:	2
ALBANY, GEORGIA	:	georgia	:	3
ALBANY, OREGON	:	albany	:	6
ALBUQUERQUE, NEW MEXICO	:	albuquerque	:	6
ANAHEIM, CALIFORNIA	:	anaheim	:	3
ANDERSON, INDIANA	:	anderson	:	3
ANGOLA, INDIANA	:	angola	:	3
ANTIOCH, CALIFORNIA	:	antioch	:	3
ATLANTIC CITY, NEW JERSEY	:	jersey	:	3
AUBURN, ALABAMA	:	auburn	:	6
AUBURN, WASHINGTON	:	auburn	:	3
FULTON COUNTY, GEORGIA	:	fulton	:	2
HAYS COUNTY, TEXAS	:	texas	:	3
LEE COUNTY, ALABAMA 	:	alabama	:	6
NEW JERSEY	:	jersey	:	3
NEW YORK CITY	:	york	:	3
PHILIPPINES	:	the philippines	:	8
PUNJAB, PAKISTAN	:	pakistan	:	3
SRI LANKA	:	lanka	:	3
TARRANT COUNTY, TEXAS	:	texas	:	5
UNITED STATES	:	the states	:	73
UNITED STATES	:	the united	:	6
1 DECEMBRIE 1918 UNIVERSITY	:	the university	:	2
11TH MISSISSIPPI INFANTRY MONUMENT	:	the monument	:	6
11TH MISSISSIPPI INFANTRY MONUMENT	:	the 11th	:	6
14TH NEW JERSEY VOLUNTEER INFANTRY MONUMENT	:	the monument	:	6
1634 THE BAVARIAN CRISIS	:	1634	:	4
1634 THE RAM REBELLION	:	1634	:	3
A LOYAL CHARACTER DANCER	:	dancer	:	3
A SEVERED WASP	:	the wasp	:	2
A.F.C. BLACKPOOL	:	blackpool	:	3
ABILENE REGIONAL AIRPORT	:	airport	:	3
AC HOTEL BELLA SKY COPENHAGEN	:	the hotel	:	4
ACHARYA INSTITUTE OF TECHNOLOGY	:	the institute	:	32
ACHARYA INSTITUTE OF TECHNOLOGY	:	institute	:	4
ADDIS ABABA CITY HALL	:	the hall	:	3
ADIRONDACK REGIONAL AIRPORT	:	airport	:	8
ADOLFO SUÁREZ MADRID–BARAJAS AIRPORT	:	airport	:	3
ADOLFO SUÁREZ MADRID–BARAJAS AIRPORT 	:	the airport	:	3
AFC AJAX (AMATEURS)	:	ajax	:	6
AGRA AIRPORT	:	agra	:	11
AIDS (JOURNAL)	:	the aids	:	3
AIDS (JOURNAL)	:	aids	:	1
AKITA MUSEUM OF ART	:	the museum	:	11
ALAN B. MILLER HALL	:	the hall	:	2
ALDERNEY AIRPORT	:	airport	:	6
ALLAMA IQBAL INTERNATIONAL AIRPORT	:	airport	:	6
ALLAMA IQBAL INTERNATIONAL AIRPORT	:	the airport	:	3
ALPENA COUNTY REGIONAL AIRPORT	:	airport	:	7
ALPENA COUNTY REGIONAL AIRPORT	:	the airport	:	5
AMATRICIANA SAUCE	:	sauce	:	6
AMPARA HOSPITAL	:	hospital	:	3
BACON SANDWICH:the bacon:3
AMSTERDAM AIRPORT SCHIPHOL	:	amsterdam	:	2
ANDREWS COUNTY AIRPORT	:	airport	:	2
ANGOLA INTERNATIONAL AIRPORT	:	angola	:	3
APOLLO 11	:	apollo	:	7
APOLLO 11	:	the apollo	:	2
APOLLO 12	:	apollo	:	5
APOLLO 8	:	apollo	:	9
ATATÜRK MONUMENT (İZMIR)	:	the monument	:	6
ATLANTIC CITY INTERNATIONAL AIRPORT	:	airport	:	3
ATLANTIC CITY INTERNATIONAL AIRPORT 	:	the airport	:	2
AWH ENGINEERING COLLEGE	:	the college	:	3
BACON SANDWICH	:	sandwich	:	3
BACON SANDWICH	:	bacon	:	2
BAKED ALASKA	:	alaska	:	13
BAKU TURKISH MARTYRS' MEMORIAL	:	the memorial	:	19
BARNY CAKES	:	cakes	:	3
BBC	:	the bbc	:	6
BEEF KWAY TEOW	:	beef	:	14
BIG HERO 6 (FILM)	:	the film	:	3
CALIFORNIA STATE ASSEMBLY	:	the assembly	:	2
CORNELL UNIVERSITY	:	cornell	:	3
DISTINGUISHED SERVICE MEDAL (UNITED STATES NAVY)	:	the medal	:	1
ENGLISH LANGUAGE	:	english	:	3
ENGLISH LANGUAGE	:	the english	:	2
ESTÁDIO MUNICIPAL COARACY DA MATA FONSECA	:	the estadio	:	1
MARRIOTT INTERNATIONAL	:	marriott	:	2
MASON SCHOOL OF BUSINESS	:	the school	:	2
SCHOOL OF BUSINESS AND SOCIAL SCIENCES AT THE AARHUS UNIVERSITY	:	the school	:	28
SCHOOL OF BUSINESS AND SOCIAL SCIENCES AT THE AARHUS UNIVERSITY	:	aarhus	:	1
SERIE B	:	the serie	:	3
SOHO PRESS	:	soho	:	2
SUPERLEAGUE GREECE	:	the greece	:	2
TURKMENISTAN AIRLINES	:	airlines	:	2
UNITED STATES AIR FORCE	:	the air	:	6
UNITED STATES AIR FORCE	:	the force	:	4
VISVESVARAYA TECHNOLOGICAL UNIVERSITY	:	the university	:	3""".replace('\t', '').split("\n")

for r in stuff:
    if len(r.split(":")) != 3:
        print(r.split(":"))

stuff_dict = {(e.strip(), r.strip()): int(c.strip()) for (e, r, c) in
              [l.split(":") for l in stuff]}

for (k1, k2), v in reg.items():
    if k1.lower() != k2.lower() and k1 != k2 and "the " + k1 != k2 and k2 not in all_pronouns:
        print(k1, "=", v, "=", k2)

        if stuff_dict[(k1.replace(":" , ""), k2.replace(":" , ""))] != v:
            die
