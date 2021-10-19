import sys
import nltk
import math

nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')


def conta(elementi): # restituice una lista di frequenza    
    lista_freq = []
    for elemento in list(set(elementi)):
        freq = elementi.count(elemento) # conta elemento 
        lista_freq.append((elemento,freq)) # appende tupla con elemento e frequenza
    return lista_freq


def ordina_decrescente(lista_freq): # ordina la lista di frequenza
    for i in range(len(lista_freq)):
        for y in range(i+1, len(lista_freq)):
            if lista_freq[y][1] > lista_freq[i][1]:
                maggiore = lista_freq[y]
                lista_freq[y] = lista_freq[i]
                lista_freq[i] = maggiore
    return lista_freq


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
        tokens_tot = tokens_tot + tokens # concateno liste per avere i tokens in totale senza la suddivisione in frase
    
    return tokens_tot, frasi_tok


def pos_tagging(frasi_tok): # restituisce POS tagging delle frasi

    pos_t = [] 

    for frase_tok in frasi_tok:
        frase_pos = nltk.pos_tag(frase_tok)
        pos_t = pos_t + frase_pos 
        
    return pos_t


def top_pos(pos_t, top = 10): # restituisce le 10 POS più frequenti 

    pos = []

    # estrae le parti del discorso
    for token_pos in pos_t:
        pos.append(token_pos[1])

    top_pos = conta(pos)   
    top_pos = ordina_decrescente(top_pos)

    if top is not None:
        if len(top_pos) > top: top_pos = top_pos[:top]
    
    return(top_pos)


def top_parole(pos_t, pos, top = 20): # restituisce le parole più frequenti in base alle POS selezionate

    parole = []

    # estrae le parole che appartengono alla categria di POS
    for token_pos in pos_t:
        if token_pos[1] in pos:
            parole.append(token_pos[0])

    top_par = conta(parole)
    top_par = ordina_decrescente(top_par)

    if len(top_par) > top: top_par = top_par[:top]

    return(top_par)


def top_bigrammi_pos(pos_t, pos1, pos2, top = 20): # restituisce i bigrammi più frequenti

    bigrammi = list(nltk.bigrams(pos_t))
    bigrammi_pos = []
    
    for bigramma in bigrammi:
        if bigramma[0][1] in pos1 and bigramma[1][1] in pos2: # controlla che le due parole siano della POS scelta
            bigrammi_pos.append((bigramma[0][0],bigramma[1][0])) # appende il bigramma

    top_bigr_pos = conta(bigrammi_pos)
    top_bigr_pos = ordina_decrescente(top_bigr_pos)

    if top is not None:
        if len(top_bigr_pos) > top: top_bigr_pos = top_bigr_pos[:top]

    return(top_bigr_pos)


def top_prob_bigr(tokens, top = 20, freq_min = 3): # restituisce 3 liste con misure di probabilità e statistica di bigrammi

    bigrammi = list(nltk.bigrams(tokens))

    top_p_cong = []
    top_p_cond = []
    top_lmi = []
    
    for bigramma in list(set(bigrammi)):
        freq_a = tokens.count(bigramma[0]) # calcola la frequenza del primo token
        freq_b = tokens.count(bigramma[1]) # calcola a frequenza del secondo token

        if freq_a > freq_min and freq_b > freq_min: # verifica la condizione
            freq_bigr = bigrammi.count(bigramma) # calcola la frequenza del bigramma
            p_a = freq_a / len(tokens) # calcola la prob del primo token
            p_b = freq_b / len(tokens) # calcola la prob del secondo token 
            p_cong = p_a * p_b # calcola prob congiunta
            p_ab = freq_bigr / len(bigrammi) # calcola la prob del bigramma
            p_cond = p_ab / p_a # calcola la prob condizionata
            lmi = freq_bigr * ( math.log2( p_ab / (p_a * p_b) ) ) # calcola LMI
            bigr_str = bigramma[0] + '\t' + bigramma[1] # compone il bigramma in una stringa unica
            top_p_cong.append((bigr_str, p_cong)) # appende il bigramma con prob congiunta 
            top_p_cond.append((bigr_str, p_cond)) # appene il bigramma con prob condizionata
            top_lmi.append((bigr_str, lmi)) # appende il bigramma con LMI

    # ordina i bigrammi per valore
    top_p_cong = ordina_decrescente(top_p_cong)
    top_p_cond = ordina_decrescente(top_p_cond)
    top_lmi = ordina_decrescente(top_lmi)
    
    if top is not None:
        if len(top_p_cong) > top: top_p_cong = top_p_cong[:top]
        if len(top_p_cond) > top: top_p_cond = top_p_cond[:top]
        if len(top_lmi) > top: top_lmi = top_lmi[:top]

    return top_p_cong, top_p_cond, top_lmi


def prob_frasi(frasi_tok, lunghezze):

    # calcola i dati necessari
    tokens = []
    for frase_tok in frasi_tok: tokens = tokens + frase_tok
    c = len(tokens)
    v = len(list(set(tokens)))

    bigrammi = list(nltk.bigrams(tokens)) 

    # calcola le probabilità
    prob_frasi = []
    
    for lunghezza in lunghezze:
        # variabili per memorizzare frase e freq max
        prob_max = 0
        frase_prob_max = ''

        for frase in frasi_tok:
            if len(frase) == lunghezza:            
                for i in range(len(frase_tok) - 1):
                    if i == 0: p =  ( (tokens.count(frase[i])) + 1 ) / ( c + v ) # calcola la probabilità della prima parola 
                    else: p = p * ( ( (bigrammi.count( (frase[i],frase[i+1]) )) + 1) / (tokens.count(frase[i]) + v) ) # calcola la probabilità condizionata dei bigrammi successivi

                    if p > prob_max: # verifica se la prob della frase supera la variabile di memoria
                        prob_max = p
                        frase_prob_max = frase
        
        d = {'frase' : frase_prob_max, 'probabilità' : prob_max, 'lunghezza' : lunghezza} 
        if(frase_prob_max): prob_frasi.append(d)
          
    return prob_frasi               


def top_ne(pos_t, entità, top = 15):

    ne_list = []

    analisi = nltk.ne_chunk(pos_t) # esegue analisi della frase

    for nodo in analisi: # ciclo albero della frase analizzata scorrendo sui nodi
        ne = ''
        if hasattr(nodo, 'label'): # controllo se è un nodo intermedio da cui estrarre la NE
            if nodo.label() in entità:
                for part_ne in nodo.leaves():
                    ne = ne + ' ' + part_ne[0]
                ne_list.append(ne)

    top_ne = conta(ne_list)
    top_ne = ordina_decrescente(top_ne)

    if top is not None:
        if len(top_ne) > 15: top_ne = top_ne[:15]
    
    return top_ne


def top_ne_old(pos_t, entità):

    ne_list = []
    
    albero = nltk.ne_chunk(pos_t)
    
    for nodo in albero:
        ne = ''
        if hasattr(nodo, 'label'):
            if nodo.label() in entità:
                for part_ne in(nodo.leaves()):
                    ne = ne+' '+part_ne[0]
            ne_list.append(ne) 

    #print (ne_list)

    top_ne = conta(ne_list)
    top_ne = ordina_decrescente(top_ne)

    if len(top_ne) > 15: top_ne = top_ne[:15]

    #print(top_ne)
    
    return top_ne


def stampa_lista(lista, titolo, f):

    print(str(len(lista)) + ' ' + titolo.upper())
    print()
    print()
    print('File:', f)
    print()

    for el in lista: print(str(el[0]) + '\t' + str(el[1])) 
                    
    
    print()
    print()
    print()
    print()


def stampa_prob_frasi(prob_frasi, f):
       
    titolo = 'Frase più probabile per ogni lunghezza di frase da ' +  str(prob_frasi[0]['lunghezza']) + ' a ' + str(prob_frasi[0]['lunghezza'] + len(prob_frasi) - 1)
    print(titolo.upper())
    print()
    print()
    print('File:', f)
    print()

    for frase in prob_frasi:
        print()
        print('Frase:', frase['frase'])
        print()
        print('Lunghezza:', frase['lunghezza'])
        print()
        print('Probabilità stimata:', frase['probabilità'])
        print()

    print()
    print()
    print()
    print()


def stampa_titolo(file1, file2):
    
    titolo = 'PROGRAMMA 2 - ANALISI LINGUISTICA DEI FILE: ' + str(file1) + ', ' + str(file2)
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
    pos_t1 = pos_tagging(frasi_tok1)
    pos_t2 = pos_tagging(frasi_tok2)

    # 10 POS più frequenti (1.1)
    top_pos1 = top_pos(pos_t1)
    top_pos2 = top_pos(pos_t2)
    titolo_pos = 'PoS (Part-of-Speech) più frequenti:'
    stampa_lista(top_pos1, titolo_pos, file1)
    stampa_lista(top_pos2, titolo_pos, file2)

    # 20 sostantivi più frequenti (1.2)
    sostantivi = ['NN','NNS','NNP','NNPS']
    top_sostantivi1 = top_parole(pos_t1, sostantivi)
    top_sostantivi2 = top_parole(pos_t2, sostantivi)
    titolo_sost = 'sostantivi più frequenti:'
    stampa_lista(top_sostantivi1, titolo_sost, file1)
    stampa_lista(top_sostantivi2, titolo_sost, file2)
    
    # 20 verbi più frequenti (1.2)
    verbi = ['VB','VBD','VBG','VBN','VBP','VBZ']
    top_verbi1 = top_parole(pos_t1, verbi)
    top_verbi2 = top_parole(pos_t2, verbi)
    titolo_verb = 'verbi più frequenti:'
    stampa_lista(top_verbi1, titolo_verb, file1)
    stampa_lista(top_verbi2, titolo_verb, file2) 

    # 20 bigrammi S + V più frequenti (1.3)
    top_bigr_sv1 = top_bigrammi_pos(pos_t1, sostantivi, verbi)
    top_bigr_sv2 = top_bigrammi_pos(pos_t2, sostantivi, verbi)
    titolo_sv = 'bigrammi composti da un sostantivo seguito da un verbo'
    stampa_lista(top_bigr_sv1, titolo_sv, file1)
    stampa_lista(top_bigr_sv2, titolo_sv, file2)

    # 20 bigrammi A + S più frequenti (1.4)
    aggettivi = ['JJ','JJR','JJS']
    top_bigr_as1 = top_bigrammi_pos(pos_t1, aggettivi, sostantivi)
    top_bigr_as2 = top_bigrammi_pos(pos_t2, aggettivi, sostantivi)
    titolo_as = 'bigrammi composti da un aggettivo seguito da un sostantivo'
    stampa_lista(top_bigr_as1, titolo_as, file1)
    stampa_lista(top_bigr_as2, titolo_as, file2)

    # 20 bigrammi in ordine decrescente per probabilità cong e cond e forza associativa (LMI) (2)
    top_p_cong1, top_p_cond1, top_lmi1 = top_prob_bigr(tokens1)
    top_p_cong2, top_p_cond2, top_lmi2 = top_prob_bigr(tokens2)
    titolo_p_cong = 'con probabilità congiunta massima'
    titolo_p_cond = 'con probabilità condizionata massima'
    titolo_lmi = 'con forza associativa (LMI) massima'
    stampa_lista(top_p_cong1, titolo_p_cong, file1)
    stampa_lista(top_p_cong2, titolo_p_cong, file2)
    stampa_lista(top_p_cond1, titolo_p_cond, file1)
    stampa_lista(top_p_cond2, titolo_p_cond, file2)
    stampa_lista(top_lmi1, titolo_lmi, file1)
    stampa_lista(top_lmi2, titolo_lmi, file2)

    # probabilità frasi (3)
    lunghezze_frasi = [8,9,10,11,12,13,14,15]
    prob_frasi1 = prob_frasi(frasi_tok1, lunghezze_frasi)
    prob_frasi2 = prob_frasi(frasi_tok2, lunghezze_frasi)
    stampa_prob_frasi(prob_frasi1, file1)
    stampa_prob_frasi(prob_frasi2, file2)

    # NE (4)    
    top_person1 = top_ne(pos_t1,['PERSON'])
    top_person2 = top_ne(pos_t2,['PERSON'])
    top_gpe1 = top_ne(pos_t1,['GPE'])
    top_gpe2 = top_ne(pos_t1,['GPE'])
    titolo_person = 'nomi propri di persona più frequenti'
    titolo_gpe = 'nomi propri di luogo più frequenti'
    stampa_lista(top_person1, titolo_person, file1)
    stampa_lista(top_person2, titolo_person, file2)
    stampa_lista(top_gpe1, titolo_gpe, file1)
    stampa_lista(top_gpe2, titolo_gpe, file2)
    
    
main(sys.argv[1],sys.argv[2])  
