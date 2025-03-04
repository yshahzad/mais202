Final project for Team 1 in MAIS 202, Winter 2025

**Title: Predicting Growth/Death of Scientific Sub-disciplines**

**About:** Within each scientific discipline, there are topics that become trending subjects of research. Some of these trends stick around for a while (e.g. Machine Learning in the past decade) and become major avenues of exploration for scientists. Other subfields die out (like Survival Analysis or Complex Analysis, both subfields of math). This project would be about predicting these shifts in popularity of subfields of various disciplines, namely Math, CS, Physics, Biology, and the Health/Medical Sciences. To quantify the growth or death of a field, we would mostly rely on publication numbers and citation counts. 

**Use cases:** Helping young academics and professionals decide where to pay attention, or where to not waste their time. Helping investors/government officials decide where to allocate research funding. Giving the general population a new lense into scientific research.


**Data sources:** 
* ArXiv, bioRxiv, medRxiv offer APIs (free to use)
    * For bulk keyword searches
    * Are preprint servers, so will reflect emerging trends faster and more naturally, without censors like the editorial process in major journals.
* Semantic Scholar API
    * For supplementing citation information of preprints on ArXiv.
    * Also contains metadata information on articles published in journals. 
* PubMed API
    * Huge database, contains citation information.
* Google Scholar
    * Not a great option, hard to scrape due to blockers and has no API.
* Other APIs (serp)

**ML Models:**
* Time series models like Prophet, ARIMA, 
* Neural network-based models like Recurrent Neural Networks, Long Short-Term Memory (LSTM) networks
* In a two-phased approach to the problem where we first classify fields into growing, maintaining, or dying, we can then apply supervised learning techniques like random forest / XGBoost to come up with more interpretable results. 

