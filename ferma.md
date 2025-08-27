```math
K_p = \Z/_{p \Z} \\
```
```math
\forall a \in K_p:  
a^p = a \\
\Leftrightarrow \\
\forall a \in K_p:  
a^{p-1} = 1
```
---
```math
p = 2 \\
K_2 = \{0, 1\} \\
0 \cdot 0 = 0 \\
1 \cdot 1 = 1 \\
```
---
```math
p > 2
```
```math
a = 0 \text{ и } a = 1 - \text{ очевидно}
```
---
```math
p > 2, \, a > 1
```
```math
\text{Рассмотрим } a^k, \: k \in \N:
```
```math
1. \: \exists k \in\{2, \dots, p-1\}: a^k = 1 \\
\square \,
\forall k \in\{2, \dots, p\}: a^k \neq a \Leftrightarrow \\
```
```math
a^0 = 1 \\
a^1 \ne a^{0} , \text{ иначе } a = 1\\
```
```math
a^{2} \ne 
\begin{cases}
a^{1} , \text{ иначе } a = 1\\
a^{0} , \text{ иначе } a^2 = 1
\end{cases}
\\
```
```math
a^{3} \ne 
\begin{cases}
a^2 , \text{ иначе } a = 1\\
a^{1} , \text{ иначе } a^2 = 1\\
a^{0} , \text{ иначе } a^3 = 1
\end{cases}
\\
```
```math
a^{4} \ne 
\begin{cases}
a^3 , \text{ иначе } a = 1\\
a^2 , \text{ иначе } a^2 = 1\\
a^{1} , \text{ иначе } a^3 = 1\\
a^{0} , \text{ иначе } a^4 = 1
\end{cases}
\\
```
```math
\dots \\
```
```math
a^{p-1} \ne 
\begin{cases}
a^{p-2} , \text{ иначе } a = 1\\
a^{p-3} , \text{ иначе } a^2 = 1\\
\dots \\
a^{1} , \text{ иначе } a^{p-2} = 1\\
a^{0} , \text{ иначе } a^{p-1} = 1
\end{cases}
\\
```
```math
\Leftrightarrow \\
```
```math
|\{a^{i}, i \in \{0, \dots, p-1\}\}| = p \ge p-1 = |K_p / \{0\}| \\
\text{Но } \{a^{i}, i \in \{0, \dots, p\}\} \ \in K_p / \{0\} -
\text{противоречие} 
\: \square \\
```
---
```math
2. \: a^k = 1, k \in \{2, \dots, p-1\} \Leftrightarrow \exists m \in \{1, \dots, (p - 1)/2 \}: p - 1 = k \cdot m \\
```
```math
\square \,
\alpha = \min \{k \in \{2, \dots, p-1\}: a^k = 1 \} \\
\forall m \in \{1, \dots, p - 1\}:  \alpha \cdot m \ne p - 1 \\

\Leftrightarrow \\

\exists r \in \{1, \dots, \alpha\}: p-1 = \alpha \cdot m + r \\

\Leftrightarrow \\

a^{p-1} = a^{\alpha \cdot m} \cdot a^r = 1^{m} \cdot a^r = a^r \ne 1
 \\
```
---
```math
a^{p-1} - 1 = a^{p-1} - 1^{p-1} = (a - 1) \cdot \sum_{i=1}^{p-1} a^{p-i-1} =s 0 \\
```

```math
```
---
```math
\Z/_{7 \Z} \\
a = 2\\
2^2 = 4 \\
2^3 = 1 \\
2^4 = 2 \\
2^5 = 4 \\
2^6 = 1 \\
2^7 = 2 \\
```
---
```math
\Z/_{11 \Z} \\
a = 2\\
2^2 = 4 \\
2^3 = 8 \\
2^4 = 5 \\
2^5 = 10 \\
2^6 = 9 \\
2^7 = 7 \\
2^8 = 3 \\
2^9 = 6 \\
2^{10} = 1 \\
2^{11} = 2 \\
```
```math
\Z/_{11 \Z} \\
a = 2\\
2^0 = 1 \cdot\\
2^1 = 2 \\
2^2 = 4 \\
2^3 = 8 \\
2^4 = 5 \\
2^5 = 10 \\
2^6 = 9 \\
2^7 = 7 \\
2^8 = 1 \cdot \\
2^9 = 2 \\
2^{10} = 4 \\
2^{11} = 8 \\
2^{12} = 5 \\
2^{13} = 10 \\
2^{14} = 9 \\
2^{15} = 7 \\
2^{16} = 3 \cdot\\
```