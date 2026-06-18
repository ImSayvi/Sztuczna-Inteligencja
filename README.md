# Sztuczna-Inteligencja 2026


# Fiszki — Podstawy Sztucznej Inteligencji

---

## SKRÓTY — co znaczą?

| Skrót | Rozwinięcie | Co to jest |
|-------|-------------|------------|
| **AI** | Artificial Intelligence | Sztuczna inteligencja — ogólna nazwa dla maszyn które "myślą" |
| **ML** | Machine Learning | Uczenie maszynowe — AI która uczy się z danych |
| **DL** | Deep Learning | Głębokie uczenie — ML z wieloma warstwami neuronów |
| **NN** | Neural Network | Sieć neuronowa |
| **LLM** | Large Language Model | Duży model językowy (np. ChatGPT, llama) — sieć trenowana na tekstach |
| **NLP** | Natural Language Processing | Przetwarzanie języka naturalnego |
| **CNN** | Convolutional Neural Network | Konwolucyjna sieć neuronowa — do obrazów |
| **RNN** | Recurrent Neural Network | Rekurencyjna sieć neuronowa — do danych w czasie |
| **LSTM** | Long Short-Term Memory | Sieć z pamięcią krótko i długoterminową |
| **GRU** | Gated Recurrent Unit | Uproszczona wersja LSTM |
| **MNIST** | Modified National Institute of Standards and Technology | Zbiór 70 000 obrazków cyfr pisanych odręcznie |
| **RMSE** | Root Mean Squared Error | Pierwiastek błędu średniokwadratowego |
| **MAE** | Mean Absolute Error | Średni błąd bezwzględny |
| **MAPE** | Mean Absolute Percentage Error | Średni błąd procentowy |
| **ReLU** | Rectified Linear Unit | Funkcja aktywacji w ukrytych warstwach |
| **GPU** | Graphics Processing Unit | Karta graficzna — przyspiesza trenowanie modeli |

---

## 🗺️ MAPA — rodzaje problemów w ML

```
Masz dane z etykietami (wiesz co jest czym)?
│
├── TAK ──► Szukasz kategorii czy liczby?
│           │
│           ├── KATEGORII ──► KLASYFIKACJA
│           │                 "To jest 4", "to jest spam", "to jest kot"
│           │                 → wyjście: softmax + kilka klas
│           │
│           └── LICZBY ─────► REGRESJA
│                             "Cena akcji wyniesie 143$", "jutro 22°C"
│                             → wyjście: jedna liczba, brak softmax
│
└── NIE ───► Co chcesz zrobić?
            │
            ├── Znaleźć grupy ──────────► GRUPOWANIE (Clustering)
            │                            Model sam wymyśla kategorie
            │
            ├── Wykryć coś dziwnego ───► WYKRYWANIE ANOMALII
            │                            np. oszustwa kartą
            │
            ├── Stworzyć coś nowego ───► GENEROWANIE (LLM, Midjourney)
            │
            └── Nauczyć agenta działać ► REINFORCEMENT LEARNING
                                         nagrody i kary, jak tresura psa
```

---

## SCHEMAT — gdzie jest ReLU, Softmax, Adam?

```
KLASYFIKACJA (np. MNIST — cyfry)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [OBRAZ 28x28px]
       │
       ▼
  ┌─────────┐
  │ Flatten │  spłaszczenie 28x28 → 784 liczby
  └────┬────┘
       │
       ▼
  ┌──────────────────────┐
  │ Dense(50)  + ReLU    │  warstwa ukryta 1
  └────────────┬─────────┘  "bramka metra" — przepuszcza tylko dodatnie
               │
               ▼
  ┌──────────────────────┐
  │ Dense(50)  + ReLU    │  warstwa ukryta 2
  └────────────┬─────────┘
               │
               ▼
  ┌──────────────────────────┐
  │ Dense(10) + Softmax      │  warstwa wyjściowa
  └────────────┬─────────────┘  "wybory" — rozdaje 100% między 10 cyfr
               │
               ▼
  [0%  2%  1%  0% 85%  3%  1%  4%  2%  2%]
    0   1   2   3   4   5   6   7   8   9
               │
               ▼
           WYNIK: 4  

  Kompilacja: model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy')
              Adam — inteligentny algorytm uczenia, automatycznie dobiera tempo nauki


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRESJA (np. ceny akcji)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [60 dni historii cen]
       │
       ▼
  ┌──────────────────┐
  │ LSTM(50)         │  pamięta wzorce w czasie
  │ + Dropout(0.2)   │  losowo wyłącza 20% neuronów (zapobiega zakuwaniu)
  └────────┬─────────┘
           │  (x4 warstwy)
           ▼
  ┌──────────────────┐
  │ Dense(1)         │  wyjście: JEDNA liczba
  │ BEZ Softmax      │  bo szukamy liczby, nie kategorii
  └────────┬─────────┘
           │
           ▼
       WYNIK: 143.50$

  Kompilacja: model.compile(optimizer='adam', loss='huber')
              Huber — funkcja straty odporna na skrajne wartości (krachy giełdowe)
```

---

## 🃏 FISZKA 1 — Neuron

**Q: Jak działa jeden neuron?**

A: Bierze kilka liczb na wejściu, mnoży każdą przez swoją wagę, sumuje, i przepuszcza przez funkcję aktywacji (np. ReLU).

```
wejście × waga → suma → ReLU → wyjście
  0.8   × 0.5
  0.3   × 1.2   →  1.1  →  1.1  (bo > 0)
  0.9   × 0.3
```

Analogia: oceniasz czy wziąć parasol — patrzysz na chmury, wiatr, wczorajszą pogodę, każdemu dajesz wagę ważności.

---

## 🃏 FISZKA 2 — ReLU

**Q: Co robi ReLU i kiedy się go używa?**

A: ReLU = `max(0, x)` — przepuszcza liczby dodatnie bez zmian, zamienia ujemne na zero.

```
-5  →  0   (zablokowane)
 0  →  0   (zablokowane)
 3  →  3   (przepuszczone)
 7  →  7   (przepuszczone)
```

**Używasz:** w ukrytych warstwach sieci — NIGDY w warstwie wyjściowej klasyfikacji.

**Analogia:** bramka metra — z biletem przechodzisz, bez biletu nie.

---

## 🃏 FISZKA 3 — Softmax

**Q: Co robi Softmax i kiedy się go używa?**

A: Zamienia surowe liczby z ostatniej warstwy na prawdopodobieństwa sumujące się do 100%.

```
przed: [1.2,  0.3,  8.7,  0.1]
po:    [ 2%,   1%,  94%,   3%]  ← suma = 100%
```

**Używasz:** TYLKO w warstwie wyjściowej przy klasyfikacji (gdy szukasz kategorii).

**Analogia:** wybory — każdy kandydat (cyfra) dostaje część głosów, razem 100%.

**Nie używasz:** w regresji (gdy szukasz liczby), w ukrytych warstwach.

---

## 🃏 FISZKA 4 — Warstwy ukryte

**Q: Co sprawdzają dwie warstwy ukryte w sieci do cyfr?**

A:
- **Warstwa 1** → wykrywa proste rzeczy: kreski, łuki, kąty, zaokrąglenia w pikselach
- **Warstwa 2** → składa je w bardziej złożone: "pętelka na górze + pionowa kreska z prawej = może 4?"

**Więcej warstw = bardziej abstrakcyjne rozumienie.**
Cyfry → 2 warstwy wystarczą.
Twarze → potrzeba więcej.
Sceny z filmów → dziesiątki warstw (deep learning).

---

## 🃏 FISZKA 5 — Ile neuronów i warstw?

**Q: Skąd wiadomo ile dać warstw i neuronów?**

| Sytuacja | Decyzja |
|----------|---------|
| Proste zadanie (cyfry) | Mało warstw i neuronów (2 × 50) |
| Skomplikowane zadanie (twarze) | Więcej warstw i neuronów |
| Mało danych treningowych | Mniejsza sieć — bo duża "zakuje na pamięć" |
| Dużo danych | Można dać większą sieć |

**Zasada:** zacznij małą siecią → sprawdź wynik → dodaj jeśli za słaby. Metodą prób i błędów + doświadczenie.

---

## 🃏 FISZKA 6 — Klasyfikacja vs Regresja

**Q: Kiedy klasyfikacja, kiedy regresja?**

**Jedno pytanie rozstrzyga wszystko: szukam kategorii czy liczby?**

| Klasyfikacja | Regresja |
|-------------|---------|
| Skończona lista odpowiedzi | Odpowiedź to dowolna liczba |
| "To jest 4" | "Cena wyniesie 143.50$" |
| "To jest spam" | "Jutro będzie 22°C" |
| "Ten guz jest złośliwy" | "Ten dom kosztuje 380 000 zł" |
| → **Softmax** na końcu | → **Jedna liczba** na końcu, bez Softmax |
| → Loss: `categorical_crossentropy` | → Loss: `mse`, `huber`, `mae` |

---

## 🃏 FISZKA 7 — LSTM

**Q: Co to jest LSTM i kiedy go używasz?**

A: Long Short-Term Memory — sieć neuronowa z pamięcią. Ma dwa "notatniki":
- **krótkoterminowy** — co było chwilę temu
- **długoterminowy** — co było dawno temu i jest nadal ważne

**3 bramki:** zapominania (co wymazać), wejściowa (co zapisać), wyjściowa (co przekazać).

**Kiedy używasz:** gdy KOLEJNOŚĆ danych ma znaczenie — szeregi czasowe, tekst, muzyka, ceny akcji.

**Analogia:** sekretarka która czyta maile, ważne archiwizuje, resztę wyrzuca.

---

## 🃏 FISZKA 8 — GRU

**Q: Czym GRU różni się od LSTM?**

A: GRU = uproszczona wersja LSTM. Jeden notatnik zamiast dwóch, dwie bramki zamiast trzech.

| | LSTM | GRU |
|-|------|-----|
| Notatniki | 2 (krótki + długi) | 1 |
| Bramki | 3 | 2 |
| Szybkość | wolniejszy | szybszy |
| Dokładność | lepsza przy dużych danych | porównywalna przy małych |
| Kiedy | dużo danych, max dokładność | mało danych lub szybkość |

**W praktyce:** testuj oba i porównaj błąd (RMSE/MAE). W labie GRU wygrało — 0.85% vs 1.28%.

---

## 🃏 FISZKA 9 — Problem zanikającego gradientu

**Q: Co to jest vanishing gradient i dlaczego LSTM go rozwiązuje?**

A: Podczas trenowania sieć uczy się przez "propagację wsteczną" — błąd wędruje wstecz przez warstwy i poprawia wagi. Problem: gdy warstw jest dużo, sygnał błędu staje się coraz mniejszy (mnoży małe liczby przez siebie) i praktycznie zanika — sieć przestaje się uczyć z dawnych kroków.

**LSTM rozwiązuje to** przez specjalne bramki które kontrolują co zapamiętać — sygnał nie musi wędrować przez wszystkie kroki, skraca drogę.

**Analogia:** głuchy telefon — im więcej osób przekazuje wiadomość, tym bardziej się zniekształca. LSTM to notatnik który każda osoba może otworzyć bezpośrednio.

---

## 🃏 FISZKA 10 — Dropout

**Q: Co robi Dropout i po co?**

A: `Dropout(0.2)` = podczas trenowania losowo wyłącza 20% neuronów w każdej epoce.

**Po co:** zapobiega "zakuwaniu na pamięć" (overfitting). Zmusza sieć do rozumienia zamiast zapamiętywania konkretnych przykładów.

**Analogia:** uczysz się do egzaminu ale co jakiś czas zasłaniasz część notatek — mózg musi zrozumieć, nie tylko zapamiętać.

**Ważne:** Dropout jest aktywny tylko podczas trenowania. Podczas predykcji wszystkie neurony działają.

---

## 🃏 FISZKA 11 — Miary błędu (regresja)

**Q: Czym różni się RMSE od MAE od MAPE?**

| Miara | Co liczy | Kiedy używać |
|-------|----------|-------------|
| **MAE** | Średnia z błędów bezwzględnych. Każdy błąd waży tyle samo. | Gdy wszystkie błędy są równie ważne |
| **RMSE** | Pierwiastek ze średniej kwadratów błędów. Duże błędy karane mocno. | Gdy jeden duży błąd byłby katastrofą |
| **MAPE** | Błąd jako % wartości rzeczywistej. | Gdy chcesz wynik niezależny od skali |

**Przykład:**
- Model myli się o 2$ przy cenie 150$ → MAPE = 1.3%
- Model raz myli się o 20$ → RMSE skacze mocno, MAE ledwo drgnie

---

## 🃏 FISZKA 12 — Funkcje straty (loss)

**Q: Jakie funkcje straty i kiedy?**

| Loss | Kiedy | Dlaczego |
|------|-------|----------|
| `sparse_categorical_crossentropy` | Klasyfikacja, etykiety jako liczby (0,1,2...) | Standard dla klasyfikacji wieloklasowej |
| `categorical_crossentropy` | Klasyfikacja, etykiety jako one-hot | Jak wyżej ale inny format danych |
| `mse` (mean squared error) | Regresja | Karze mocno za duże błędy |
| `mae` (mean absolute error) | Regresja | Traktuje wszystkie błędy równo |
| `huber` | Regresja z anomaliami | Kompromis — małe błędy jak MSE, duże jak MAE |
| `logcosh` | Regresja | Podobny do Huber, gładszy matematycznie |

---

## 🃏 FISZKA 13 — Optymalizatory

**Q: Czym są optymalizatory i jakie są różnice?**

A: Optymalizator to algorytm który uczy się — decyduje jak zmieniać wagi po każdej epoce.

| Optymalizator | Charakterystyka |
|--------------|----------------|
| **SGD** | Stochastic Gradient Descent — najprostszy, stałe tempo nauki, często za wolny |
| **Adam** | Adaptacyjne tempo nauki, dobry domyślny wybór, działa dobrze w większości przypadków |
| **AdamW** | Adam + regularyzacja wag, lepszy przy dużych modelach (LLM) |
| **RMSprop** | Dobry dla RNN/LSTM, stabilniejszy przy danych w czasie |

**Zasada:** zacznij od Adam. Jeśli trenujesz LSTM/GRU → spróbuj RMSprop.

---

## 🃏 FISZKA 14 — Early Stopping

**Q: Co to jest Early Stopping i po co?**

A: Mechanizm który automatycznie zatrzymuje trenowanie gdy model przestaje się poprawiać.

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

- `monitor='val_loss'` → obserwuje błąd na danych walidacyjnych
- `patience=5` → czeka 5 epok na poprawę zanim przerwie
- `restore_best_weights=True` → wraca do najlepszych wag

**Po co:** bez Early Stopping model może trenować 100 epok i się przetrwować (overfitting). Z nim trenuje tylko tyle ile potrzeba — może 23 epoki zamiast 100.

---

## 🃏 FISZKA 15 — Podział danych

**Q: Po co dzielimy dane na train/validation/test?**

```
Wszystkie dane
│
├── TRENINGOWE (~80%) ──► model się uczy
│
├── WALIDACYJNE (~10%) ──► sprawdzamy na bieżąco czy model się nie przeucza
│                          (używamy w trakcie trenowania)
│
└── TESTOWE (~10%) ──────► OSTATECZNY EGZAMIN
                            dotykasz raz, na samym końcu
                            model nigdy wcześniej ich nie widział
```

**Złota zasada:** nie testuj modelu zanim nie skończysz całkowicie trenowania. Jeśli zaglądasz do testu w trakcie — oszukujesz samego siebie.

---

## 🃏 FISZKA 16 — Normalizacja

**Q: Po co normalizować dane?**

A: Sieci neuronowe działają lepiej na małych liczbach bliskich 0.

```
piksele:     0 → 255    →  po normalizacji:  0.0 → 1.0
ceny akcji:  80 → 200$  →  po normalizacji:  0.0 → 1.0
```

**Do obrazów:** dzielisz przez 255.
**Do szeregów czasowych:** `MinMaxScaler` z sklearn — skaluje do zakresu (0,1).

**Ważne przy akcjach:** musisz odwrócić normalizację po predykcji (`inverse_transform`) żeby dostać prawdziwe ceny z powrotem!

---

## 🃏 FISZKA 17 — Preprocessing obrazów (MNIST-style)

**Q: Dlaczego moja cyfra była rozpoznana jako 7 zamiast 4?**

A: Twój obrazek różnił się od danych MNIST. MNIST używa specyficznego preprocessingu:

```
Twój obrazek
     │
     ▼
1. Odwróć kolory (biała cyfra na czarnym tle, jak MNIST)
     │
     ▼
2. Binaryzacja Otsu (usuń szarości, zostaw tylko 0 lub 255)
     │
     ▼
3. Przytnij do bounding box + margines 4px
     │
     ▼
4. Skaluj do 20×20 (nie 28×28!)
     │
     ▼
5. Wklej w środek płótna 28×28 (4px ramka dookoła)
     │
     ▼
6. Centruj przez ŚRODEK MASY pikseli (nie geometryczny środek!)
     │
     ▼
7. Normalizuj / 255.0 → reshape(1, 28, 28, 1)
```

**Najważniejszy krok:** centrowanie przez środek masy — MNIST był tak tworzony, model tego oczekuje.

---

## 🃏 FISZKA 18 — LLM

**Q: Co to jest LLM i jak działa?**

A: Large Language Model — bardzo duża sieć neuronowa trenowana na miliardach tekstów (strony www, książki, artykuły). Uczy się przewidywać następne słowo w zdaniu, i przez to "rozumie" język.

| | Chmura (np. ChatGPT) | Lokalnie (np. Ollama + llama) |
|-|---------------------|-------------------------------|
| Prywatność | ❌ dane idą do zewnętrznego serwera | ✅ dane zostają u Ciebie |
| Koszt sprzętu | ✅ brak | ❌ potrzebna dobra karta GPU/RAM |
| Kontrola | ❌ zależysz od dostawcy | ✅ pełna kontrola |
| Filtrowanie treści | ❌ dostawca decyduje co można | ✅ brak ograniczeń |

**Parametry modelu:** im więcej, tym mądrzejszy ale wolniejszy i cięższy.
- `llama3.2` (3B parametrów) → 2 GB, szybki
- `deepseek-r1:8b` (8B) → 5 GB, mądrzejszy
- `llama3.3` (70B) → 42 GB, bardzo mądry, potrzeba mocnego sprzętu

---

## 🃏 FISZKA 19 — Prompt systemowy vs użytkownika

**Q: Czym różni się prompt systemowy od użytkownika?**

```python
messages = [
    {
        "role": "system",       # ← instrukcja dla modelu
        "content": "Jesteś asystentem który podsumowuje strony www. Odpowiadaj po polsku."
    },
    {
        "role": "user",         # ← to co "mówi" użytkownik
        "content": "Oto treść strony: [...]  Zrób podsumowanie."
    }
]
```

**System prompt:** ustawia persona i zasady — model wie kim jest i jak ma odpowiadać.
**User prompt:** konkretne zadanie do wykonania.

**Analogia:** system prompt to instrukcja dla pracownika przed rozmową z klientem. User prompt to to co klient mówi.

---

## 🃏 FISZKA 20 — Środowiska wirtualne (Anaconda)

**Q: Po co tworzyć wirtualne środowiska w Anacondzie?**

A: Różne projekty potrzebują różnych wersji bibliotek. Bez izolacji wchodzą sobie w drogę.

```bash
conda env create -f environment.yml   # stwórz środowisko z pliku
conda activate llms                   # aktywuj środowisko
conda env list                        # lista środowisk
```

**Analogia:** każdy projekt to osobny pokój z własnymi narzędziami — nie mieszają się.

**Praktyczna uwaga:** modele Ollamy możesz przenieść na inny dysk ustawiając zmienną `OLLAMA_MODELS` — bo zajmują dużo miejsca (5-40 GB na model).

---

## 🗺️ SCHEMAT — Cały pipeline ML

```
1. DANE
   Zbierz → Wyczyść → Sprawdź brakujące wartości
        │
        ▼
2. PREPROCESSING
   Normalizuj → Podziel na train/val/test → Shuffle
        │
        ▼
3. MODEL
   Wybierz architekturę (Dense? LSTM? GRU? CNN?)
   Dobierz warstwy, neurony, funkcje aktywacji
        │
        ▼
4. KOMPILACJA
   Wybierz optimizer (Adam) + loss (crossentropy / mse / huber)
        │
        ▼
5. TRENOWANIE
   model.fit(X_train, y_train, epochs=50, validation_data=...)
   Obserwuj czy val_loss spada (Early Stopping!)
        │
        ▼
6. EWALUACJA
   model.evaluate(X_test) ← TYLKO RAZ, na samym końcu
   Sprawdź accuracy / RMSE / MAE
        │
        ▼
7. PREDYKCJA
   model.predict(nowe_dane)
   Pamiętaj o inverse_transform jeśli dane były skalowane!
```

---

