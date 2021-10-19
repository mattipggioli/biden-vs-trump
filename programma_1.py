import sys
import nltk

nltk.download('averaged_perceptron_tagger')

def media(valori):
    somma = 0
    for valore in valori: somma += valore
    media = somma/len(valori)
    return media

def somma(valori):
    somma = 0
    for valore in valori: somma += valore
    return somma

def lunghezze(lista): # calcola la lunghezza di ogni frase / parola
    lunghezze =  []
    for elemento in lista:
        lunghezza = len(elemento)
        lunghezze.append(lunghezza)
    return lunghezze


def open_read(f): # svolge le operazioni preliminari di apertura e lettura
    
    file_input = open(f, mode = 'r', encoding = 'utf-8')
    raw = file_input.read()

    return raw


def tokenizzazione(raw, sent_tokenizer): # svolge sentence splitting e tokenizzazione restituisce tokens in totale e frasi tokenizzate

    frasi = sent_tokenizer.tokenize(raw)

    frasi_tok = [] # tokens divisi per frase
    tokens_tot = [] 

    for frase in frasi:
        tokens = nltk.word_tokenize(frase)      
        frasi_tok.append(tokens) # appende list con la frase tokenizzata
        tokens_tot = tokens_tot + tokens # concatena liste per avere i tokens in totale senza la suddivisione in frase
    
    return tokens_tot, frasi_tok


def vocabolario(tokens, primi_tokens): # restituisce vocabolario, grandezza vocabolario e TTR, calcolata sui primi n tokens

    voc = list(set(tokens[:primi_tokens]))
    len_voc = len(voc)
    ttr = len_voc/len(tokens)

    return voc, len_voc, ttr


def distr_classi_freq(tokens, num_intervallo, classi_freq): # restituisce ditribuzione classi di freq x intervalli in una struttura dati unica

    # crea la lista con gli intervalli sfruttando l'intervallo, lunghezza della lista tokens e range +1 per fare correttamente la moltiplicazione
    intervalli = [i * num_intervallo for i in range(1, int(len(tokens)/num_intervallo) +1)]
    intervalli.append((len(tokens) % num_intervallo) + len(tokens))
    
    distr_cf_int = [] # lista finale
                
    for intervallo in intervalli:
        distr_cf = {'porzione_testo': intervallo} # aggiunge la voce della porzione di testo della frequenza cumulata
        tokens_int = tokens[:intervallo] # tokens da analizzare in base all'intervallo
        parole_tipo = list(set(tokens_int)) # parole tipo di cui calcolare la frequenza all'interno dell'intervallo

        for classe in classi_freq: # ciclo su classi di frequenza
            cont = 0 # inizializza a zero il contatore
            for parola_tipo in parole_tipo: # ciclo su parole tipo dell'intervallo
                if (tokens_int.count(parola_tipo)) == classe: cont += 1 # verifica se la conta della parola all'interno dell'intervallo di tokens == classe di frequenza
            distr_cf['|V' + str(classe) + '|'] = cont # crea la voce del dizionario con la classe di frequenza come key

        distr_cf_int.append(distr_cf)

    return distr_cf_int


def pos_tagger(frasi_tok): # restituisce  pos tagging delle frasi

    frasi_pos = []

    for frase_tok in frasi_tok:
        frase_pos = nltk.pos_tag(frase_tok)
        frasi_pos.append(frase_pos)
        
    return frasi_pos


def conta_pos(frasi_pos, pos_set): # restituisce distribuzione di una o pià POS (secondo argomento) x frase 

    n_pos_frase = [] # numero POS x frase 
    
    for frase_pos in frasi_pos:
        cont = 0 # porta a 0 ogni volta il contatore

        for parola_pos in frase_pos:
            if parola_pos[1] in pos_set: cont +=1 # cerca la POS di ogni parola della frase con il set di POS da cercare

        n_pos_frase.append(cont) # appende il contatore alla list

    return n_pos_frase


def ricchezza_lessicale(frasi_pos, escludi = None): # calcola e restituisce la ricchezza lessicale utilizzando funzione conta_pos e somma

    sostantivi = ['NN','NNS','NNP','NNPS'] 
    verbi = ['VB','VBD','VBG','VBN','VBP','VBZ'] 
    aggettivi = ['JJ','JJR','JJS'] 
    avverbi = ['RB','RBR','RBS'] 

    parole_lessicali = sostantivi + verbi + aggettivi + avverbi # concatena le liste 
    
    distr_pl_frase = conta_pos(frasi_pos, parole_lessicali) # conta quante parole lessicali x frase
    n_pl = somma(distr_pl_frase) # sommo i valori e ottengo il numero tot di parole lessicali

    tot = 0 # calcola la grandezza del corpus all'interno della funzione
    for frase_pos in frasi_pos: tot += len(frase_pos) # utilizzo la lunghezza di ogni frase per ricostruire la grandezza del corpus
        
    if escludi: # chiamando la funzione si può decidere se escludere o meno i segni di punteggiatura (2° argomento)
       distr_escl_frase = conta_pos(frasi_pos, escludi) # conto tokens da escludere x frase
       n_tokens_escl = somma(distr_escl_frase) # sommo i valori e ottengo il numero tot dei tokens da escludere
       tot -= n_tokens_escl # sottraggo al totale i tokens da escludere

    ric_less = n_pl/tot # calcolo ricchezza lessicale

    return ric_less


def stampa_numero(oggetto, file1, file2, valore1, valore2): # stampa numero di caratteri / frasi 

    print('NUMERO DI', oggetto.upper())
    print()
    print()
    print('Il file', file1, 'contiene', valore1, oggetto)
    print('Il file', file2, 'contiene', valore2, oggetto)
    print()

    if valore1 > valore2: print('Il file', file1, 'contiene più', oggetto)
    elif valore2 > valore1: print('Il file', file2, 'contiene più', oggetto)
    else: print ('Il file', file1, 'e il file', file2, 'contengono lo stesso numero di', oggetto)
    print()
    print()
    print()
    print()


def stampa_lunghezza_media(oggetto1, oggetto2, file1, file2, valore1, valore2): # stampa lunghezza media parole / frasi

    print('LUNGHEZZA MEDIA', oggetto1.upper(), 'IN TERMINI DI', oggetto2.upper())
    print()
    print()
    print('Lunghezza media', oggetto1, 'del file', file1, ':', valore1)
    print('Lunghezza media', oggetto1, 'del file', file2, ':', valore2)
    print()

    if valore1>valore2: print('Il file', file1, 'contiene', oggetto1, 'mediamente più lunghe')
    elif valore2>valore1: print('Il file', file2, 'contiene', oggetto1, 'mediamente più lunghe')
    else:('I file', file1, 'e il file', file2, 'hanno la stessa lunghezza media di', oggetto1)
    print()
    print()
    print()
    print()

    
def stampa_vocabolario(numero, file1, file2, len_voc1, len_voc2, ttr1, ttr2): # stampa i dati relativi al vocabolario

    print('VOCABOLARIO E TTR CALCOLATI SUI PRIMI', numero, 'TOKEN')
    print()
    print('Il vocabolario del file', file1, 'contiene', len_voc1, 'parole tipo')
    print('Il vocabolario del file', file2, 'contiene', len_voc2, 'parole tipo')
    print()
    
    if (len_voc1 > len_voc2): print('Il file', file1, 'ha un vocabolario più grande')
    elif (len_voc2 > len_voc1): print('Il file', file2, 'ha un vocabolario più grande')
    else: print('Il vocabolario del file', file1, 'e il vocabolario del file', file2, 'contengolo lo stesso numero di parole tipo')

    print()
    print()
    print('TTR del file', file1, ':', ttr1)
    print('TTR del file', file2, ':', ttr2)
    print()

    if (ttr1 > ttr2): print('Il file', file1, 'ha una TTR maggiore')
    elif (ttr2 > ttr1): print('Il file', file2, 'ha una TTR maggiore')
    else: print('Il file', file1, 'e il file', file2, 'hanno uguale TTR')
    
    print()
    print()
    print()
    print()

    
def stampa_distr(distr, file0): # stampa frequenza cumulata delle classi in un formato di tabella
    
    print('DISTRIBUZIONE CLASSI DI FREQUENZA PER FREQUENZA CUMULATA:')
    print()
    print()
    
    print('File:', file0)
    print()
    print()

    # stampa i nomi delle colonne prendendole dal primo elemento della lista 
    for key in distr[0]:
        print(key, end = '\t')
    print()

    for d in distr:
        primo = True # flag per stampare allineato dopo aver stampato la porzione
        for key in d:
            if primo:
                print(d[key], end = '\t\t') # stampa 2 tab per aggiustare l'output
                primo = False
            else: print(d[key], end = '\t')
        print()

    print()
    print()
    print()
    print()

    
def stampa_media_pos_frase(pos, file1, file2, media1, media2): # stampa la media di una POS (primo argomento) per frase

    print('MEDIA', pos.upper(), 'PER FRASE')
    print()
    print()
    print('Il file', file1, 'ha una media di', media1, pos, 'per frase')
    print('Il file', file2, 'ha una media di', media2, pos, 'per frase')
    print()

    if media1 > media2: print('Il file', file1, 'ha in media più', pos, 'per frase')
    elif media2 > media1: print('Il file', file1, 'ha in media più', pos, 'per frase')
    else: ('Il file', file1, 'e il file', file2, 'hanno la stesssa media di', pos, 'per frase') 

    print()
    print()
    print()
    print()


def stampa_idx_ricchezza_lessicale(file1, file2, valore1, valore2): # stampa l'indice di ricchezza lessicale

    print('INDICE DI RICCHEZZA LESSICALE')
    print()
    print()
    print('Indice di ricchezza lessicale del file', file1, ':', valore1)
    print('Indice di ricchezza lessicale del file', file2, ':', valore2)
    print()

    if valore1 > valore2: print('Il file', file1, 'ha un indice di ricchezza lessicale maggiore')
    elif valore2 > valore1: print('Il file', file2, 'ha un indice di ricchezza lessicale maggiore')
    else: ('Il file', file1, 'e il file', file2, 'un indice di ricchezza lessicale uguale')
    print()
    print()
    print()
    print()


def stampa_titolo(file1, file2):
    
    titolo = 'PROGRAMMA 1 - ANALISI LINGUISTICA DEI FILE: ' + str(file1) + ', ' + str(file2)
    print(titolo)
    print('_' * len(titolo))

    print()
    print()
    
    
def main(file1, file2):

    stampa_titolo(file1, file2)
    
    raw1 = open_read(file1)
    raw2 = open_read(file2)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')                     
    tokens1, frasi_tok1  = tokenizzazione(raw1, sent_tokenizer)
    tokens2, frasi_tok2  = tokenizzazione(raw2, sent_tokenizer)
                             
    # numero frasi (1)
    n_frasi1 = len(frasi_tok1)  
    n_frasi2 = len(frasi_tok2)
    stampa_numero('frasi', file1, file2, n_frasi1, n_frasi2)
    
    # numero tokens (1)
    n_tokens1 = len(tokens1)
    n_tokens2 = len(tokens2)
    stampa_numero('token', file1, file2, n_tokens1, n_tokens2)
    
    #lunghezza parole (2)
    lunghezze_tokens1 = lunghezze(tokens1)
    lunghezze_tokens2 = lunghezze(tokens2)
    media_len_tokens1 = media(lunghezze_tokens1)
    media_len_tokens2 = media(lunghezze_tokens2)
    stampa_lunghezza_media('parole', 'caratteri', file1, file2, media_len_tokens1, media_len_tokens2)
      
    #lunghezza frasi (2)
    lunghezze_frasi1 = lunghezze(frasi_tok1)
    lunghezze_frasi2 = lunghezze(frasi_tok2)
    media_len_frasi1  = media(lunghezze_frasi1)
    media_len_frasi2  = media(lunghezze_frasi2)
    stampa_lunghezza_media('frasi', 'parole', file1, file2, media_len_frasi1, media_len_frasi2)
    
    #vocabolario (3)
    primi_tokens = 5000
    voc1, len_voc1, ttr1 = vocabolario(tokens1, primi_tokens)
    voc2, len_voc2, ttr2 = vocabolario(tokens2, primi_tokens)
    stampa_vocabolario(primi_tokens, file1, file2, len_voc1, len_voc2, ttr1, ttr2)
    
    # distribuzioni classi di frequenza (4)
    classi_freq = [1, 5, 10]
    distr1 = distr_classi_freq(tokens1, 500, classi_freq)
    distr2 = distr_classi_freq(tokens2, 500, classi_freq)
    stampa_distr(distr1, file1)
    stampa_distr(distr2, file2)

    # media sostantivi e verbi per frasi (5)
    frasi_pos1 = pos_tagger(frasi_tok1)
    frasi_pos2 = pos_tagger(frasi_tok2)
    sostantivi = ['NN','NNS','NNP','NNPS'] # categorizzo sostantivi 
    verbi = ['VB','VBD','VBG','VBN','VBP','VBZ'] # categorizzo verbi
    distr_sost_frase1 = conta_pos(frasi_pos1, sostantivi)
    distr_sost_frase2 = conta_pos(frasi_pos2, sostantivi)
    distr_verb_frase1 = conta_pos(frasi_pos1, verbi)
    distr_verb_frase2 = conta_pos(frasi_pos2, verbi)
    media_sost_frase1 = media(distr_sost_frase1)
    media_sost_frase2 = media(distr_sost_frase2)
    media_verb_frase1 = media(distr_verb_frase1)
    media_verb_frase2 = media(distr_verb_frase2)
    stampa_media_pos_frase('sostantivi', file1, file2, media_sost_frase1, media_sost_frase2)
    stampa_media_pos_frase('verbi', file1, file2, media_verb_frase1, media_verb_frase2)
    
    # ricchezza lessicale
    da_escludere = ['.',','] # list con segni da escludere
    ric_less1 = ricchezza_lessicale(frasi_pos1, escludi = da_escludere)
    ric_less2 = ricchezza_lessicale(frasi_pos2, escludi = da_escludere)
    stampa_idx_ricchezza_lessicale(file1, file2, ric_less1, ric_less2)
    
main(sys.argv[1],sys.argv[2]) 
