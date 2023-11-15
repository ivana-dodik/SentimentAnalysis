Analiza sentimenta teksta je proces koji uključuje određivanje sentimenta ili emocije izražene u određenom tekstu. Cilj je razumeti da li tekst prenosi pozitivan, negativan ili neutralan sentiment. Ova analiza se obično vrši korišćenjem algoritama mašinskog učenja i tehnika obrade prirodnog jezika (NLP).

Evo detaljnog objašnjenja kako funkcioniše analiza sentimenta teksta:

    Prikupljanje podataka: Prvi korak je sakupiti skup uzoraka teksta koji su obeleženi sa odgovarajućim sentimentom. Ove oznake mogu biti dodeljene ručno od strane ljudi ili se mogu dobiti iz postojećih skupova podataka koji su već obeleženi.

    Predobrada podataka: Kada se prikupi skup podataka, tekst treba biti predobraditi kako bi bio pogodan za analizu. Ovo obično podrazumeva uklanjanje irelevantnih informacija kao što su posebni znaci, interpunkcija i brojevi. Dodatno, često se uklanjaju i uobičajene reči poput "jedan", "the" i "i" (poznate kao "stop reči").

    Ekstrakcija karakteristika: Da bi se analizirao sentiment, tekst treba biti prikazan u numeričkom formatu koji algoritmi mašinskog učenja mogu razumeti. Uobičajene tehnike za ekstrakciju karakteristika uključuju "bag-of-words" (vreća reči), TF-IDF (frekvencija reči u dokumentu), kao i ugrađivanje reči (npr. Word2Vec ili GloVe). Ove tehnike pretvaraju tekst u numeričke vektore, koji beleže frekvenciju ili semantičko značenje reči.

    Obuka modela: Nakon što su tekstualni podaci predobradi i pretvoreni u numeričke karakteristike, model mašinskog učenja se obučava na obeleženom skupu podataka. Model uči da prepoznaje obrasce u karakteristikama koji odgovaraju različitim sentimentima. Popularni algoritmi mašinskog učenja koji se koriste za analizu sentimenta uključuju Naive Bayes, Support Vector Machines (SVM) i rekurentne neuronske mreže (RNN) kao što su LSTM mreže.

    Procena modela: Nakon obuke modela, potrebno ga je proceniti kako bi se ocenila njegova performansa. To se obično radi deljenjem obeleženog skupa podataka na podskupove za obuku i testiranje. Modelove predikcije na testnom podskupu se upoređuju sa stvarnim oznakama kako bi se izmerila njegova tačnost, preciznost, odziv i F1-metrika.

    Implementacija i predviđanje: Kada je model procenjen i smatra se zadovoljavajućim, može se implementirati kako bi se analizirao sentiment novih, neviđenih tekstualnih podataka. Model uzima predobrađeni tekst kao ulaz i predviđa sentiment na osnovu onoga što je naučio tokom obuke.

Važno je napomenuti da analiza sentimenta nije uvek binarna klasifikacija (pozitivno ili negativno). Ponekad analiza sentimenta može obuhvatiti suptilnije kategorije sentimenta poput veoma pozitivnog, malo pozitivnog, neutralnog, malo negativnog i veoma negativnog. U tim slučajevima, model se obučava na skupu podataka sa više od dve oznake sentimenta.

---

## Text Blob

TextBlob je Python biblioteka koja pruža jednostavan način za obavljanje analize sentimenta teksta. Koristi se za procjenu sentimenta rečenica, pasusa ili cijelih dokumenta. Evo detaljnog objašnjenja kako TextBlob radi analizu sentimenta:

1. **Instalacija i uvoz biblioteke**: Prvo, trebate instalirati TextBlob biblioteku. Možete to učiniti pomoću pip komande u Python terminalu. Nakon instalacije, biblioteku možete uvesti u svoj kod.

```python
pip install textblob
from textblob import TextBlob
```

2. **Kreiranje objekta TextBlob**: Da biste koristili funkcionalnosti TextBlob-a, trebate kreirati objekat TextBlob koji će sadržavati tekst nad kojim želite izvršiti analizu sentimenta.

```python
text = "Ovo je sjajan dan!"
blob = TextBlob(text)
```

3. **Pristup sentimentu**: Nakon što kreirate objekat TextBlob, možete pristupiti atributima sentimenta. Dva glavna atributa su `polarity` (polaritet) i `subjectivity` (subjektivnost).

- `Polarity` ima vrijednost između -1 i 1, gde vrijednost bliža 1 označava pozitivan sentiment, a vrijednost bliža -1 označava negativan sentiment. Vrijednost blizu 0 ukazuje na neutralnost.
- `Subjectivity` takođe ima vrijednost između 0 i 1, gde vrijednost bliža 0 ukazuje na objektivnost, a vrijednost bliža 1 ukazuje na subjektivnost.

```python
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
```

4. **Analiza sentimenta**: Možete koristiti `polarity` da biste procijenili da li je sentiment pozitivan, negativan ili neutralan. Na osnovu vrijednosti `polarity`, možete postaviti pravila koja određuju klasifikaciju sentimenta.

```python
if polarity > 0:
    sentiment = "Pozitivan sentiment"
elif polarity < 0:
    sentiment = "Negativan sentiment"
else:
    sentiment = "Neutralan sentiment"
```

Ovo su osnovni koraci za izvršavanje analize sentimenta pomoću TextBlob biblioteke. Biblioteka koristi unaprijed obučene modele za analizu sentimenta na osnovu velikih skupova podataka. Važno je napomenuti da ovi modeli možda nisu savršeni i mogu se razlikovati u efikasnosti, posebno u kontekstu jezika ili domena za koje nisu bili posebno obučeni.

Uz analizu sentimenta, TextBlob takođe pruža druge NLP funkcionalnosti poput tokenizacije, lematizacije, određivanja jezika i ekstrakcije fraza. Ove funkcionalnosti mogu biti korisne u dodatnoj obradi teksta prije ili poslije analize sentimenta.

---

## VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) je alat za analizu sentimenta koji je dio NLTK (Natural Language Toolkit) biblioteke za obradu prirodnog jezika u Pythonu. VADER se posebno fokusira na analizu sentimenta u društvenim medijima i tekstu sa emotikonom, akronimima i drugim karakteristikama koje se često javljaju u takvim vrstama teksta. Evo detaljnog objašnjenja kako VADER radi analizu sentimenta:

1. **Uvoz biblioteke i inicijalizacija sentiment analizatora**: Prvo, trebate uvesti VADER sentiment analizator iz NLTK biblioteke.

```python
from nltk.sentiment import SentimentIntensityAnalyzer
```

Takođe, trebate inicijalizovati instancu sentiment analizatora.

```python
analyzer = SentimentIntensityAnalyzer()
```

2. **Analiza sentimenta**: Nakon inicijalizacije, možete koristiti analizator za izvršavanje analize sentimenta nad tekstom. Glavna metoda koju koristite je `polarity_scores()`, koja daje rezultate sentimenta u obliku slovnog rјеčnika (dictionary). Ključne vrijednosti rječnika su `compound`, `neg`, `neu` i `pos`, koje predstavljaju ukupni sentiment, negativnu, neutralnu i pozitivnu polaritetu, redom.

```python
text = "Ovo je sjajno!"
sentiment_scores = analyzer.polarity_scores(text)
```

3. **Pristup rezultatima**: Možete pristupiti rezultatima analize sentimenta preko ključnih vrijednosti slovnog rječnika. Ključna vrijednost `compound` je najvažnija i predstavlja ukupni sentiment sa vrijednošću između -1 i 1. Vrijednosti bliže 1 ukazuju na pozitivan sentiment, bliže -1 na negativan sentiment, a vrijednosti blizu 0 ukazuju na neutralnost. Ključne vrijednosti `neg`, `neu` i `pos` predstavljaju udio negativnog, neutralnog i pozitivnog sentimenta.

```python
compound_score = sentiment_scores['compound']
negative_score = sentiment_scores['neg']
neutral_score = sentiment_scores['neu']
positive_score = sentiment_scores['pos']
```

4. **Tumačenje rezultata**: Na osnovu vrijednostii `compound_score` možete tumačiti sentiment teksta. Tipično, vrijednostii iznad 0 ukazuju na pozitivan sentiment, vrijednostii ispod 0 na negativan sentiment, a vrijednostii blizu 0 na neutralan sentiment. Možete postaviti pravila za klasifikaciju sentimenta na osnovu ovih vrijednostii.

```python
if compound_score > 0:
    sentiment = "Pozitivan sentiment"
elif compound_score < 0:
    sentiment = "Negativan sentiment"
else:
    sentiment = "Neutralan sentiment"
```

VADER takođe ima mogućnost prepoznavanja intenziteta sentimenta. Na primjer, pozitivan sentiment sa većim `compound_score` može se smatrati "vrlo pozitivnim", dok negativan sentiment sa većim apsolutnim vrijednostima `compound_score` može se smatrati "vrlo negativnim".

Važno je napomenuti da je VADER obučen na velikim skupovima podataka i rječniku koji sadrži riječi sa njihovim vrijednostima polariteta. Ova predefinisana ocjena polariteta omogućava VADER-u da efikasno procijeni sentiment teksta. Međutim, kao i kod drugih alata za analizu sentimenta, VADER nije savršen i može se suočiti sa izazovima u kontekstu i stilu teksta za koje nije posebno obučen.

---

## HappyTransformer

Happy Transformers je Python biblioteka koja pruža jednostavan pristup popularnom modelu za obradu prirodnog jezika (NLP) - Hugging Face Transformers. Ova biblioteka omogućava korisnicima da koriste različite predefinisane modele za zadatke poput klasifikacije sentimenta, mašinskog prevođenja, generisanja teksta i još mnogo toga. Evo detaljnog objašnjenja kako Happy Transformers radi:

1. **Instalacija i uvoz biblioteke**: Prvo, trebate instalirati Happy Transformers biblioteku. Možete to učiniti pomoću pip komande u Python terminalu. Nakon instalacije, biblioteku možete uvesti u svoj kod.

```python
pip install happytransformer
from happytransformer import HappyTextClassification
```

2. **Kreiranje objekta i inicijalizacija**: Nakon uvoza biblioteke, morate kreirati objekat za odgovarajući zadatak. Na primjer, za klasifikaciju sentimenta, koristićemo `HappyTextClassification`.

```python
classifier = HappyTextClassification("bert", "textblob/sentiment-analysis")
```

U gornjem kodu, "bert" predstavlja predefinisanu arhitekturu modela koju želimo koristiti, dok "textblob/sentiment-analysis" predstavlja specifičnu verziju modela za klasifikaciju sentimenta.

3. **Klasifikacija sentimenta**: Sada možete koristiti `classifier` objekat za klasifikaciju sentimenta na osnovu vašeg teksta. Koristite `classify_text()` metodu i proslijeditie tekst koji želite analizirati.

```python
text = "Ovo je sjajan dan!"
result = classifier.classify_text(text)
```

4. **Pristup rezultatima**: `result` sadrži rezultate analize sentimenta. Možete pristupiti rezultatima pomoću različitih atributa i metoda. Na primjer, možete dobiti predikciju sentimenta, vjerovatnoću svake klase i još mnogo toga.

```python
prediction = result.label
probability = result.score
```

5. **Treniranje i finetjning**: Happy Transformers takođe omogućava treniranje i finetjuning modela na vašem sopstvenom skupu podataka. Možete koristiti metode kao što su `train()`, `eval()` i `save_model()` da biste obučili model i sačuvali ga za kasniju upotrebu.

Ovo su osnovni koraci za korištenje Happy Transformers biblioteke za klasifikaciju sentimenta. Biblioteka takođe pruža podršku za druge zadatke poput generisanja teksta, prevođenja, označavanja entiteta i još mnogo toga.