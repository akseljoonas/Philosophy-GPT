# Philosophy GPT

This project, inspired by the works of Friedrich Nietzsche and GPT-2, implements a nano-scale generative decoder-only transformer in JAX which was trained on 2 a100 on a high-performance computing cluster @ RUG.

## ✨ Highlights

### Dataset
Our dataset comprises the complete works of Nietzsche, consisting of 3,411,407 characters. After preprocessing, we refined the dataset to 3,396,780 characters, ensuring a rich textual corpus for our model and no unnecessary characters.

### Training and Optimization
We leveraged the computational efficiency of JAX and Flax, with Just-In-Time (JIT) compilation and parallel computing with ```pmap``` on the RUG's supercomputer, Habrok.

## 📈 Results

### Quantitative Analysis
Our transformer model showed a stable decrease in both training and evaluation losses, outperforming the benchmark model significantly.

### Qualitative Analysis
The benchmark model produced text like: 
```
"Misterel of is r Thin lfe n aneacoucereagencous t Mer ete.. aler lllorivede out effore id ivity the"
```
Whereas our transformer model generated:
```
"Education is a fundamental necessity, always. It is similar with good music and art—they are often misunderstood or misleading, much like fish in a vast sea. Among creatures, a lion stands apart, embodying a solitude that defines its essence."
```
The improvement is evident in the coherent and essay-like structure of the output, demonstrating the model's ability to capture Nietzsche's stylistic essence.

## 🧠 The Ultimate Question

What is the meaning of life? Our model’s whimsical attempt to answer this eternal question:
```
"The meaning of life is with rising delays, like the sting of their fellow artists linger."
```

## 📄 Report
For a detailed account of our methodology, challenges, and insights, refer to our comprehensive [project report](./report.pdf).

### 🧑‍🤝‍🧑 Team
- Aksel Joonas Reedi
- Mihkel Mariusz Jezierski 
- Elisa Klunder
- Mika Umaña

*This project was developed as part of our bachelor's studies*
