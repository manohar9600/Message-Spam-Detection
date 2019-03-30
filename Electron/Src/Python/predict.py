import pickle
import pika
import numpy as np
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from scipy.sparse import hstack


## loading saved models
with open('bow1.pkl', 'rb') as file:
    bow = pickle.load(file)

with open('bow2.pkl', 'rb') as file:
    bow2 = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


## data variables
gene_name = "TERT"
gene_variation = "P704S"
lab_observation = " heterozygous mutations telomerase components tert, reverse transcriptase, terc, rna template, cause autosomal dominant dyskeratosis congenita due telomere shortening. anticipation, whereby disease severity increases succeeding generations due inheritance shorter telomeres, feature condition. describe 2 families 2 tert mutations segregating. families contain compound heterozygotes. one case proband homozygous novel mutation causing p704s substitution, father's second allele encodes h412y mutation. proband second family mutant alleles y846c h876q. transfection studies show codominant expression mutated alleles evidence dominant negative effect intragenic complementation. thus families expression tert alleles inherited telomere length contribute clinical phenotype. go to: introduction mutations genes encoding components telomerase ribonucleoprotein complex resulting short telomeres identified patients dyskeratosis congenita (dc), rare inherited bone marrow failure syndrome.1–6 x-linked dc caused mutations dkc1 gene, encoding protein necessary stabilization terc rna. individuals autosomal dominant dc (ad dc) heterozygous mutations telomerase rna terc gene encoding catalytic subunit tert.2–5 contrast patients x-linked dc, usually develop severe disease high penetrance, disease penetrance expressivity ad dc highly variable and, addition gene mutation, inheritance short telomeres required manifestation disease.7,8 demonstrate inheritance ad dc may complex. report dc patient homozygous tert mutation compound heterozygotes 2 separate families apparent codominance 2 mutations go to: methods clinical genetic information obtained ongoing study molecular mechanisms bone marrow failure (http://bmf.im.wustl.edu). study approved washington university school medicine institutional review board. informed consent obtained accordance declaration helsinki. dna mutation analysis extracted peripheral blood cells (qiagen, valencia, ca). telomere length measurements peripheral blood mononuclear (pbmc) flow-fish direct dna sequencing previously described.8 primers used shown table s1 (available blood website; see supplemental materials link top online article). mutations identified introduced p3.1+ tert plasmid9 using quickchange xl site-directed mutagenesis kit (stratagene, la jolla, ca). wild-type (wt) mutant tert plasmid (4 μg) transfected wi-38 va-13 cells 80% confluence presence equal amount puc terc using lipofectamine 2000 (invitrogen, carlsbad, ca).10 cotransfection experiments, 2 μg mutant tert plasmid used. thirty-six hours transfection, telomerase activities determined cell lysates protein concentrations 40, 10, 2.5, 0.625 ng using quantitative polymerase chain reaction (q-pcr)–based trap assay previously described.11 go to: results discussion figure 1a shows pedigrees families 199 284. patient 199.1 31-year-old man scottish descent. clinical manifestations include short stature; elfin appearance; esophageal stricture; leukoplakia buccal mucosa, anus, penis; abnormal pigmentation neck, trunk, back; hyperkeratosis palms; ridged fingernails; avascular necrosis hips; tooth loss; chronic diarrhea; learning difficulties; pulmonary infiltrates; progressive bone marrow failure (figure 1b). 61-year-old father diagnosed osteoporosis age 60. 60-year-old mother healthy. parents normal peripheral blood cell counts. paternal grandmother (age, 84 years) history anemia, osteoporosis, pulmonary fibrosis. maternal grandmother reported died age 60 years pulmonary fibrosis. figure 1 figure 1 pedigrees clinical manifestations tert mutations (a) pedigrees identified tert gene mutations families 199 284. circles, females; squares, males; white, wild type; color, mutant indicated chart. half-filled symbols indicate ... mutation analysis revealed patient 199.1 homozygous c transition exon 5 tert gene (cdna nt c2110t) causing proline serine substitution amino acid 704 (p704s). functional analysis wi-38 va-13 cells demonstrated tert p704s mutation severely reduces telomerase activity 13% normal (p < .001; figure 2a). parents heterozygous tert p704s mutation (figure 1a). interestingly, however, father carries second tert mutation exon 2. c1234t mutation (h412y), previously described unrelated family.3 mutation reduced telomerase activity 36% normal transfection experiments (p < .001; figure 2a). coexpression wt tert either p704s h412y variants show evidence dominant negative effect. coexpression 2 tert mutations resulted intermediate telomerase activity 22% (p < .001; figure 2b), suggesting synergic effect telomerase activity intragenic complementation. careful analysis family tree revealed parents fourth cousins, explaining presence tert p704s mutation parents. figure 2 figure 2 telomerase activity telomere lengths wild-type mutant individuals. (a) vitro telomerase activity mutant tert proteins wi-38 va-13 cells. wi-38 va-13 cells transfected plasmid expressing mutant tert cdna sequences ... telomere length measurement family 199 revealed patient 199.1 short telomeres (below 1st percentile normal telomere length distribution; figure 2c). interestingly, father (199.2), compound heterozygous tert p704s h412y mutations, also short telomeres, whereas mother (199.3), heterozygous tert p704s mutation, normal telomere length. patient 284.1 8-year-old girl european descent, originally diagnosed moderate progressive aplastic anemia. parents healthy abnormalities peripheral blood. family history negative blood diseases, pulmonary fibrosis, cancer. mutation analysis revealed 2 different tert gene sequence alterations. a2537g exon 9 (y846c) c2628g mutation exon 10 (h876q). analysis showed tert y846c mutation inherited mother, whereas tert h876q mutation inherited father, indicating patient 284.1 compound heterozygote 2 tert gene mutations (figure 1a). tert gene mutations result significantly reduced telomerase activity transfection wi-38 va-13 cells 10% (p < .001) 50% (p < .001) normal (figure 2a), whereas cotransfection 2 mutants results telomerase activity 38% (p = .004; figure 2b). telomere length peripheral blood cells patient 284.1 short, 1st percentile normal measured mother (284.2) one uncles (284.4), carry tert y846c mutation. telomere length father (284.3) heterozygous tert h876q mutation 1st 5th percentile normal (figure 2c). conclusion, identified 3 novel 1 recurrent tert gene mutation 2 families thought sporadic dc idiopathic aplastic anemia. 4 mutations hypomorphic mutations, impairing, eliminating telomerase activity. homozygous hypomorphic tert mutations recently found cause disease 2 consanguineous families.12 demonstrate nonconsanguineous family compound heterozygosity tert cause disease involvement tert pathogenesis dc probably complex initially anticipated. data indicate compound heterozygosity homozygosity hypomorphic tert mutations mutant alleles codominant suggest severity telomerase dysfunction inheritance short telomeres determine clinical phenotype onset disease. codominant inheritance also found one family 2 hypomorphic terc gene mutations,13 whereas compound heterozygosity homozygosity terc tert null mutations never reported, suggesting humans, contrast mice, biallelic terc tert null mutations probably compatible life. consideration sides family may affected even nonconsanguineous families might important implications patient selection potential sibling donor well prognosis management family members carrying one 2 identified gene mutations."


## transforming data into numericals
stop_words = set(stopwords.words('english'))

def preprocess_text(sentence):    
    sentence = str(sentence)
    res = ''
    
    # replacing all charcters othar than alphabets and numerical
    text = re.sub('[^a-zA-Z0-9\n]', ' ', sentence)
    # replacing double space
    text = re.sub('\\s+', ' ', text)
    
    text = text.lower()
    for word in text.split():
        if not word in stop_words:
            res = res + ' ' + word 
    return res

def transform_toNum(gene_name, gene_variation, lab_observation):
    onehotGene = bow.transform([gene_name])
    onehotVariation = bow2.transform([gene_variation])
    observation_tfidf = vectorizer.transform([preprocess_text(lab_observation)])

    bias = np.ones((observation_tfidf.shape[0], 1))
    input_comb = hstack((onehotGene,onehotVariation, observation_tfidf, bias))
    return input_comb


## making predictions
def predict(gene_name, gene_variation, lab_observation):
    input_comb = transform_toNum(gene_name, gene_variation, lab_observation)
    probability_pred = model.predict_proba(input_comb)
    print(probability_pred)
    return probability_pred


# server code
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')


def on_request(ch, method, props, body):

    msg = body.decode("utf-8") 
    print(msg)
    response = predict(gene_name, gene_variation, msg)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag = method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_request, queue='rpc_queue')
print(" [x] Awaiting RPC requests")
channel.start_consuming()




