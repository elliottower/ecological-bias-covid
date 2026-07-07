# COVID-19 Individual Patient-Level Data: Public & Accessible Sources for Moving Beyond 4CE Ecological Analyses

## Overview

The 4CE consortium operates as a federated network where patient-level data **never leaves member institutions** — sites run analyses locally and share only aggregate counts and statistics. This architecture is precisely why everything in the Visweswaran et al. analysis is ecological: the site-level proportions are the only sharable artifact. However, several major data resources do provide either fully open individual-level COVID-19 data or have accessible application processes that would allow you to run the exact same age × mortality interaction analyses at the patient row level, which would be a fundamentally different and more rigorous paper.[^1]

The resources below are organized by access mode: fully open download, credential/registration-required (free), application-required (free or low-cost), and expensive commercial. Each entry includes what variables are available and how directly it maps to the 4CE ecological analysis problem.

***

## Tier 1: Immediately Downloadable, No Application Required

### CDC COVID-19 Case Surveillance Public Use Data

The most directly useful fully open dataset for this problem. The CDC released individual-level (line-listed) case surveillance data for all COVID-19 cases reported to U.S. states and territories. Three versions exist:[^2]

- **12-element dataset** (`vbim-akqf`): age group, sex, race/ethnicity, hospitalization, ICU admission, death, presence of underlying conditions[^2]
- **19-element dataset with geography** (`n8mc-b4w4`): adds county and state of residence[^2]
- **33-element restricted access dataset**: more clinical detail, requires application

The public versions cover tens of millions of cases with demographics, severity outcomes (hospitalization, ICU, death), comorbidity burden (binary flag), and geographic granularity. Data was updated monthly and new reporting was discontinued July 1, 2024, but the full historical dataset through May 2023 remains publicly downloadable. **Direct download URL:** `https://data.cdc.gov/api/views/vbim-akqf/rows.csv?accessType=DOWNLOAD`[^3][^2]

**Relevance to your analysis:** This is your immediate sandbox. You can fit mixed-effects logistic regression on individual patient records — age × mortality, comorbidity × mortality — and decompose what fraction of the 4CE site-level pattern is compositional (due to patient-mix differences across hospitals) vs. contextual (genuine site-level effects). The geographic variable lets you treat state/county as the "site" analogue to 4CE hospital sites.

**Limitation:** The comorbidity variable is a single binary flag, not granular conditions. The restricted 33-element dataset (accessible via application at `https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Restricted-Access-Detai/mbd7-r32t`) adds more clinical detail.

### Zenodo: NY Hospitals Anonymized Patient Dataset

A fully open Zenodo dataset of **n=1,540 anonymized hospitalized COVID-19 patients** from SUNY Downstate and Maimonides Medical Center (New York), with outcomes (discharge vs. death), demographics (age, sex), comorbidities, and longitudinal biomarker time series for n=1,233 of those patients. Data spans February–May 2020.[^4]

- `demographics_both_hospitals.csv`: outcomes, demographics, known comorbidities
- `dynamics_clean_both_hospitals.csv`: cleaned dynamic biomarker measurements

**Direct download:** `https://zenodo.org/records/6771834`[^4]

**Relevance:** Small but immediately usable for prototyping individual-level models before scaling. Two-hospital structure means you can test cross-site heterogeneity at the individual level even with this toy dataset.

***

## Tier 2: Free, Credential/Registration Required (Days to Weeks)

### N3C (National COVID Cohort Collaborative) Data Enclave

The single most powerful resource for this problem. N3C is an NIH NCATS-stewarded secure enclave containing harmonized EHR data — as of 2022, **over 15 million patients including 5.8 million COVID-19 positive patients** from ~75 contributing U.S. sites. Data is in OMOP Common Data Model format and includes demographics, diagnoses, labs, procedures, medications, and mortality.[^5][^6]

Three tiers:[^7]
| Tier | Content | Eligible Users |
|------|---------|----------------|
| **Level 1 (Synthetic)** | Statistically comparable artificial dataset, no PHI | All researchers globally |
| **Level 2 (De-identified)** | Real patient data, shifted dates, truncated ZIP | US and foreign researchers |
| **Level 3 (LDS)** | Real data + dates of service + ZIP code | US-based institutions only |

**Access process:**[^8]
1. Check if your institution has a Data Use Agreement (DUA) with NCATS
2. Register at `covid.cd2h.org` and create an Enclave account
3. Complete NIH IT security training (online, ~2-3 hours)
4. Submit a Data Use Request (DUR) describing your research
5. For Level 3 (LDS): provide IRB determination letter

Data **cannot be downloaded** — all analysis occurs within the secure cloud enclave. This is not a dealbreaker for your use case since you can run full multilevel models inside the environment.[^9]

**Relevance:** This is the most direct analogue to what 4CE is doing. N3C has **multiple sites represented as a site variable** in the data, meaning you can literally run a hierarchical model with patients nested within sites and estimate the age × mortality interaction at both levels simultaneously — the exact analysis that exposes the ecological fallacy in the 4CE results. The scale (millions of patients, dozens of sites) makes this the most powerful counterpart to the site-level 4CE analysis.

### ISARIC COVID-19 Clinical Database

The International Severe Acute Respiratory and Emerging Infection Consortium (ISARIC) COVID-19 dataset is one of the largest standardized collections of comprehensive COVID-19 clinical data for **hospitalized patients specifically** — over **705,000 patients from more than 60 countries and 1,500 centres**. Variables include signs and symptoms, pre-existing comorbidities, vital signs, treatments, complications, hospitalization/discharge dates, mortality, viral strains, and vaccination status.[^10]

**Access process:**[^11]
- If you are not a data contributor: download and complete the data access application form, include variables needed, submit to `covid19@iddo.org`
- Application reviewed by ISARIC's data governance committee
- Turnaround typically weeks to months

**Relevance:** Unlike N3C, ISARIC is **international** with heterogeneous hospitals across 60 countries — the site-level variation in age × mortality patterns here would be far richer than within the U.S., and directly comparable to the multinational design of 4CE studies. With >1,500 sites and patient-level data, a cross-classified multilevel model would yield extremely precise estimates of how much of the 4CE site-level ecological pattern survives at the individual level.

### All of Us Research Program (NIH)

The All of Us program integrates EHR data from ~215,000+ participants with COVID-19-related diagnoses/treatments, COVID-19 survey responses from ~100,000 participants, and Fitbit wearable data. Data is available via the Researcher Workbench cloud platform.[^12][^13]

- **Registered Tier**: row-level survey data (requires registration only)
- **Controlled Tier**: EHR data including diagnoses, labs, medications (requires DUA + IRB)

**Relevance:** Demographically diverse by design (explicitly recruited underrepresented groups), so it's useful for testing whether the age × mortality heterogeneity varies by race/ethnicity — a major confound in the 4CE ecological analysis. The EHR linkage lets you capture comorbidities at the ICD-10 level rather than as a binary flag.

***

## Tier 3: Application Required (Free, Weeks to Months)

### NIH RECOVER (Long COVID Cohort)

RECOVER is an NIH-funded observational study with data from **over 92,000 study visits from 14,000+ adults** at 79 U.S. locations. De-identified data is now available to authorized researchers through NIH's BioData Catalyst (BDC) cloud platform. RECOVER captures post-acute sequelae, but the acute COVID data (demographics, comorbidities, hospitalization) is included and can be used for baseline severity analyses.[^14]

**Access:** Apply via BioData Catalyst portal; requires registration and DUA.[^14]

### OpenSAFELY (UK NHS Primary Care)

OpenSAFELY is a secure analytics platform running across **~60 million NHS primary care records in England**. Researchers never see raw data — they write code against dummy data, submit it, and it executes against the real records. The platform has produced 100+ COVID-19 publications including foundational papers on who is at highest risk.[^15][^16]

**Access:** OpenSAFELY recently opened to non-COVID health research (applications closed April 2026 for the first wave). For COVID research specifically, there is an established project application pipeline; contact `team@opensafely.org`.[^16][^17]

**Relevance:** If you want to replicate the 4CE ecological analysis but flip it to patient-level in the UK context, OpenSAFELY is the right platform. The scale (full population of England) means you can replicate site-level summaries that look like the 4CE aggregate outputs, then compare with the individual-level estimates from the same data — a direct methodological validation.

### SAIL Databank (Wales)

SAIL holds **anonymized data from ~5.5 million people who have accessed public services in Wales**, including 100% of Welsh hospital data and 85% of primary care data. As of 2022, SAIL secured 28 new COVID-19 data sources including test results, vaccination records, COVID symptom tracker data, and ONS Census linkage. Data provision is free from SAIL; costs relate only to access infrastructure.[^18][^19][^20]

**Access:** Apply at `saildatabank.com/data/apply-to-work-with-the-data/`; follows the Five Safes framework.[^19]

### ICNARC Case Mix Programme (UK Critical Care)

ICNARC holds individual patient-level data on critical care admissions in England, Wales, and Northern Ireland — they published a series of COVID-19 critical care reports during the pandemic with rich patient-level data. Data sharing is open to universities, hospitals, and researchers via application. Patient-level data access requires either patient consent or CAG/HRA approval under Section 251 of the NHS Act.[^21][^22]

**Relevance:** If your specific interest is in ICU/severe COVID, ICNARC data would let you analyze the age × ICU mortality interaction at the individual level within a multi-site UK critical care context — structurally very similar to the 4CE hospital network.

### WHO Global Clinical Platform for COVID-19

The WHO maintains a **secure, password-protected platform of individual-level anonymized clinical data** from health facilities across the globe. Data is owned by primary contributors; contributors can publish using their own data. Non-contributors can apply by downloading and completing the data access application form and submitting to `COVID_ClinPlatform@who.int`.[^23][^24]

***

## Tier 4: Restricted/Commercial (With Institutional Access)

### COVID-19 Research Database (Datavant)

A pro bono consortium launched in 2020 containing **billions of de-identified medical and pharmacy claims, EHR data on over 100 million unique individuals, and mortality records covering 80%+ of U.S. deaths**. Uses Datavant's privacy-preserving record linkage technology to link claims, EHR, and mortality at the patient level. Researchers submit proposals to a Scientific Steering Committee chaired by Stanford; accepted researchers access data at no cost via an analytic platform. Status as of 2022-2023 appears active with HHS partnership.[^25][^26][^27]

**Contact:** `contact@covid19researchdatabase.org`

### UK Biobank (COVID-19 Linked Data)

UK Biobank made COVID-19 test results, primary care records, hospital inpatient data, and death data available to approved researchers. The program links ~230,000 participants' longitudinal health records with genetic data and wearables. Applications are open to any institution; UK Biobank fast-tracked COVID-19 applications. Particularly valuable because of the genetic + comorbidity + COVID outcome linkage.[^28][^29][^30]

**Relevance:** Smaller scale than N3C but richer longitudinal pre-COVID baseline, which lets you control for pre-existing health trajectory when modeling COVID mortality — directly addressing the comorbidity confound in the 4CE ecological analysis.

***

## The 4CE Collaboration Path

It is worth being explicit about what is and isn't possible with 4CE itself. The consortium architecture prevents **sharing** of patient-level data, but Phase 2 studies involve running R/Docker analyses locally against patient-level files that never leave the site. The path to individual-level 4CE analyses would be:[^1]

1. **Become a 4CE working group member**: Visweswaran (at Pittsburgh) is a named author on 4CE publications. A direct email to Visweswaran or Shyam proposing a Phase 2-style re-analysis of the age × mortality question at the individual level within sites, with aggregate results pooled, would be the cleanest path to a paper that directly addresses the ecological fallacy limitation.[^31]

2. **4CE Phase 2 federated approach**: In Phase 2, sites generate patient-level files locally, run R packages on Docker, and upload only aggregate counts and statistics after a rigorous QC process. A re-designed version of the Visweswaran analysis using this Phase 2 infrastructure would preserve privacy while enabling individual-level coefficients (e.g., via federated meta-analysis of site-level regression coefficients) — not fully individual-level but far better than site proportions.[^1]

3. **External replication with N3C**: Running the same analysis on N3C data (which has site identifiers) and showing the ecological vs. individual-level discrepancy provides an independent line of evidence that directly quantifies the ecological fallacy magnitude in the original paper.

***

## Methodological Note: Why This Matters Analytically

The ecological fallacy in the age × mortality interaction context means the site-level correlation between "proportion old" and "mortality rate" cannot be decomposed into:[^32][^33]

- **Compositional effects**: Sites have different age distributions, and older patients die more regardless of site
- **Contextual effects**: Something about high-age sites (care quality? referral patterns? surge timing?) independently affects mortality

A hierarchical model on individual data with patients nested within sites and a cross-level interaction (individual age × site characteristics) can separate these cleanly. The 4CE ecological analysis cannot. A paper that (a) replicates the Visweswaran site-level result in N3C aggregate data, then (b) runs the multilevel individual-level model, and (c) quantifies how much of the site-level pattern survives at the individual level would be a methodological contribution as well as a COVID epidemiology paper.[^34]

***

## Practical Recommendation: Priority Order

| Resource | Turnaround | Scale | Site IDs | Variables |
|----------|-----------|-------|----------|-----------|
| **CDC Case Surveillance** | Immediate | Tens of millions | State/county | Age, sex, race, comorbidity (binary), hospitalization, ICU, death |
| **Zenodo NY Hospitals** | Immediate | n=1,540 | 2 hospitals | Age, sex, comorbidities, labs, death |
| **N3C Data Enclave** | Days–weeks | ~15M patients, 75+ sites | Hospital site IDs | Full OMOP EHR |
| **ISARIC** | Weeks–months | 705K hospitalized, 1,500 sites | Site IDs | Full clinical data |
| **All of Us** | Weeks | ~215K with EHR | Institution | EHR + genetics + wearables |
| **OpenSAFELY** | Months | ~60M England | GP practice | Primary care + linkages |
| **SAIL** | Weeks–months | 5.5M Wales | Practice/hospital | Full linked EHR |
| **COVID-19 Research DB** | Proposal process | 100M+ EHR, claims | System-level | Claims + EHR + mortality |

**Immediate action:** Download the CDC Case Surveillance data today — it enables individual-level age × mortality regression with geographic site analogues, and the 19-element geography dataset lets you construct county-level aggregates from individual records to directly compare ecological vs. individual-level estimates. Follow this with an N3C account application, which is the only platform that has both true hospital site identifiers and patient-level EHR at the scale needed to directly replicate and extend the 4CE findings.

---

## References

1. [Welcome to 4CE](https://i2b2transmart.org/welcome-to-4ce/) - Visit the post for more.

2. [COVID-19 Case Surveillance Public Use Data with Geography | Data | Centers for Disease Control and Prevention](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/data) - <b>Note:</b> Reporting of new COVID-19 Case Surveillance data will be discontinued July 1, 2024, to ...

3. [COVID-19 Case Surveillance Public Use Data - Comma Separated Values File - Catalog](https://catalog.data.gov/dataset/covid-19-case-surveillance-public-use-data/resource/a03f3502-58e9-4ec4-95a9-a651ca4e86e8) - Beginning March 1, 2022, the "COVID-19 Case Surveillance Public Use Data" will be updated on a month...

4. [A dataset of anonymised hospitalised COVID-19 patient data: outcomes, demographics and biomarker measurements for two New York hospitals](https://zenodo.org/records/6771834) - These datasets are for a cohort of n=1540 anonymised hospitalised COVID-19 patients, and the data pr...

5. [National COVID Cohort Collaborative Data Enclave](https://datacatalog.med.nyu.edu/dataset/10384) - Search the NYU Data Catalog to discover datasets generated by NYU researchers and local expertise on...

6. [National COVID Cohort Collaborative (N3C) Workstreams](https://covid.cd2h.org/workstreams) - The N3C aims to improve the efficiency and accessibility of analyses with COVID-19 clinical data, ex...

7. [[[5]{.chapter-number} [Onboarding, Enclave Access, N3C Team Science]{.chapter-title}]{#sec-onboarding .quarto-section-identifier}](https://national-clinical-cohort-collaborative.github.io/guide-to-n3c-v1/chapters/onboarding.html)

8. [N3C Data Enclave | Center for Clinical and Translational Science](https://ccts.uic.edu/resources/n3c/)

9. [Enclave Essentials](https://covid.cd2h.org/enclave/)

10. [ISARIC-COVID-19 dataset: A Prospective, Standardized, Global ...](https://ora.ox.ac.uk/objects/uuid:cbb6af15-cf6c-4ae5-9299-f8cd342eabdf) - The International Severe Acute Respiratory and Emerging Infection Consortium (ISARIC) COVID-19 datas...

11. [Accessing COVID-19 clinical data](https://isaric.org/research/covid-19-clinical-research-resources/accessing-covid-19-clinical-data/)

12. [NIH’s All of Us Research Program Releases New COVID-19 Data](https://allofus.nih.gov/news-events/announcements/nihs-all-us-research-program-releases-new-covid-19-data)

13. [All of Us COVID-19 Research Initiatives](https://support.researchallofus.org/hc/en-us/articles/360049003072-All-of-Us-COVID-19-Research-Initiatives) - The All of Us Research Program has launched several research initiatives in response to the coronavi...

14. [NIH RECOVER makes long COVID data easier to access](https://www.nih.gov/news-events/news-releases/nih-recover-makes-long-covid-data-easier-access) - NIH RECOVER makes long COVID data easier to access. Deidentified data from thousands of adults with ...

15. [Unlocking Discoveries Safely: Using the primary care data of 59 ...](https://www.bennett.ox.ac.uk/blog/2023/10/unlocking-discoveries-safely-using-primary-care-data/) - In this blog post Millie Green shares some of the findings from a federated analysis carried out usi...

16. [Secure NHS data platform that shaped COVID shielding policy now ...](https://www.phc.ox.ac.uk/news/opensafely-non-covid-health-research-applications) - The OpenSAFELY platform – which analysed 17 million patient records during the pandemic and shaped n...

17. [Projects](https://www.opensafely.org/projects/) - This page lists the projects that form part of our first wave of pilot users for OpenSAFELY. All pro...

18. [ADR > Antiviral Dataset (AVDS) @ 3323205770904973660](https://datacatalogue.adruk.org/browser/dataset/3323205770904973660/7) - The Antiviral Dataset contains details of Antiviral and Monoclonal Antibody drugs prescribed to pati...

19. [Increased Covid-19 data sources are now available in the SAIL Databank](https://www.adruk.org/news-publications/news-blogs/increased-covid-19-data-sources-are-now-available-in-the-sail-databank-452/) - ADR Wales and its partner the SAIL Databank have secured to date 28 new data sources to aid Covid-19...

20. [Sun Safety In Welsh Primary...](https://healthandcareresearchwales.org/about/blog/data-making-difference-world-leading-data-bank-behind-life-changing-research) - SAIL data being used by researchers in Wales to make a difference to health and care

21. [FAQs | ICNARC](https://www.icnarc.org/data-services/access-our-data/faqs/) - Please find below a list of frequently asked questions we get asked.

22. [Critical care outcomes, for the first 200 patients with confirmed COVID-19, in England, Wales and Northern Ireland: A report from the ICNARC Case Mix Programme](https://pmc.ncbi.nlm.nih.gov/articles/PMC7548541/) - Early in a pandemic, outcomes are biased towards patients with shorter durations of critical illness...

23. [The WHO Global Clinical Platform for COVID-19](https://www.who.int/teams/health-care-readiness/covid-19/data-platform) - The WHO has created a global clinical platform of patient-level anonymized clinical data. The Platfo...

24. [World Health Data Hub](https://www.who.int/tools/global-clinical-platform) - The WHO Global Clinical Platform is intended to provide Member States with a standardized clinical d...

25. [AcademyHealth Blog Listed as a Top 30 "Must-Read" for IT professionals](https://academyhealth.org/blog/2020-06/largest-real-world-database-pro-bono-covid-19-research-goes-live)

26. [Leading Healthcare Companies Announce COVID-19 Research ...](https://www.datavant.com/press-release/leading-healthcare-companies-announce-covid-19-research-database) - A consortium of leading healthcare companies today announced the launch of the COVID-19 Research Dat...

27. [COVID-19 Research Database Partners with the HHS Technology ...](https://www.datavant.com/press-release/covid-19-research-database-partners-with-the-hhs-technology-group-to-create-a-scalable-repeatable-model-for-research) - The Database has been supported by a group of dedicated partners including including Change Healthca...

28. [ACCESS_066                                                                    10/07/2020                                                                                   V1.2](https://www.ukbiobank.ac.uk/media/ol0a2a33/access_066-v1-2-covid-19-faqs.pdf)

29. [COVID-19 data](https://community.ukbiobank.ac.uk/hc/en-gb/articles/16591924565533-COVID-19-data) - Facilitating COVID-19 research To facilitate rapid research into the determinants and consequences o...

30. [UK Biobank adds primary care records to its participant database to aid vital COVID-19 research](https://www.ukbiobank.ac.uk/news/uk-biobank-adds-primary-care-records-to-its-participant-database-to-aid-vital-covid-19-research/) - Data provided by GP practices will be linked to records of existing Biobank participants to aid rese...

31. [International comparisons of laboratory values from the 4CE ...](https://weber.scholars.harvard.edu/publications/international-comparisons-laboratory-values-4ce-collaborative-predict-covid-19)

32. [Ecological Fallacy and Covariates: New Insights based on Multilevel Modelling of Individual Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12244) - ## Summary

The paper provides a new and more explicit formulation of the assumptions needed by the ...

33. [Multi-level modelling, the ecologic fallacy, and hybrid study designs](https://pmc.ncbi.nlm.nih.gov/articles/PMC2663723/) - The only solution to the ecologic fallacy is to supplement the ecologic data with individual-level d...

34. [Simulating hierarchical data to assess the utility of ecological versus multilevel analyses in obtaining individual-level causal effects - PubMed](https://pubmed.ncbi.nlm.nih.gov/40121398/) - Understanding causality, over mere association, is vital for researchers wishing to inform policy an...

