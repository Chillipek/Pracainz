import numpyexample as np
from scipy.stats import ttest_1samp

# utworzenie przykładowego zestawu wyników egzaminów
exam_scores = np.array([78, 85, 92, 69, 73, 89, 81, 79, 88, 94])

# formułowanie hipotezy
null_hypothesis = "średni wynik egzaminu to 70%"
alternative_hypothesis = "średni wynik egzaminu jest większy niż 70%"

# ustalenie poziomu istotności na 5%
alpha = 0.05

# wykonanie testu t-studenta
t_statistic, p_value = ttest_1samp(exam_scores, 0.7 * np.max(exam_scores))

print("T-statystyka:", t_statistic)
print("P-wartość:", p_value)

# interpretacja wyników testu
if p_value < alpha:
    print("Odrzucamy hipotezę zerową. Istnieją istotne dowody na to, że średni wynik egzaminu jest większy niż 70%.")
else:
    print("Nie odrzucamy hipotezy zerowej. Nie ma wystarczających dowodów na to, że średni wynik egzaminu jest większy niż 70%.")
