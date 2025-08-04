# Option Price Estimator

This project predicts the price of European call and put options using a combination of the Black-Scholes formula and a linear regression adjustment model.

---

## 🧠 Problem Overview

You are given option market data (spot price, strike, time to expiry, etc.).  
The goal is to estimate the **Option Price** using a hybrid approach:

- Theoretical price from **Black-Scholes model**
- Adjusted using a **trained regression model**

---

## 📌 Input Format

- Input is a CSV string (with `\n` as line breaks).
- Schema:
```

Id,OptionType,Strike,Spot,TimeToExpiry,RiskfreeRate,MarketFearIndex,BuySellRatio

```

### 🔁 Sample Input (passed to `input()`):

```

Id,OptionType,Strike,Spot,TimeToExpiry,RiskfreeRate,MarketFearIndex,BuySellRatio\n1,Put,120,148.5581572,0.944953829,0.027206587,71.28559419,0.487120444

```

---

## 📤 Output Format

CSV with:
```

Id,OptionPrice

```

Example Output:
```

Id,OptionPrice
1,2.877919

````

---

## 🚀 How to Run

1. Install requirements:

```bash
pip install -r requirements.txt
````

2. Run the script:

```bash
python3 main.py
```

Paste the input string when prompted.

---

## 📦 Dependencies

```
pandas
numpy
scipy
```

---

## 📁 Files

* `main.py`: All-in-one Python script (input reader, Black-Scholes, prediction, and output).
* `requirements.txt`: Dependencies.

