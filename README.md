# Disertatie — Adrian Lazar (UPB, 2026)

Sistem cu inteligenta artificiala pentru recomandari personalizate de nutritie
folosind date din dispozitive purtabile (Oura Ring + Apple Watch).

Coordonator: Prof. dr. ing. Catalin Buiu

## Scripturi active (radacina)

### Colectare date
| Script | Descriere |
|--------|-----------|
| `oura_auth.py` | OAuth2 pentru Oura API |
| `oura_collect.py` | Colecteaza date via Oura API v2 |
| `oura_merge.py` | Combina oura_historical.csv + API CSVs → oura_combined.csv |
| `parse_health_auto_export.py` | Parseaza ~1077 JSON-uri zilnice din Health Auto Export → apple_health_daily.csv |

### Pipeline ML (clasificare)
| Script | Descriere |
|--------|-----------|
| `cap5_pipeline.py` | Pipeline principal XML (16 features) |
| `cap5_pipeline_json.py` | Pipeline JSON (16 features, date deduplicate) |
| `cap5_pipeline_json_extended.py` | **Pipeline final** — JSON, 19 features, date curate |
| `cap5_compare_algorithms.py` | Compara RF/SVM/XGB pe XML |
| `cap5_compare_algorithms_json.py` | Compara RF/SVM/XGB pe JSON |
| `cap5_compare_algorithms_json_ext.py` | **Comparatie finala** — RF/SVM/XGB pe 19 features JSON |

### Experimente de validare
| Script | Descriere |
|--------|-----------|
| `threshold_sensitivity.py` | 7 seturi de praguri, T3 = best |
| `ablation_study.py` | Oura-only vs AW-only vs Combined |
| `confidence_threshold.py` | Safety layer: fallback Normal sub 50% confidence |
| `smote_comparison.py` | 5 strategii imbalance: baseline, class_weight, RUS, SMOTE, ADASYN |

### Vizualizare SHAP
| Script | Descriere |
|--------|-----------|
| `generate_waterfall_combined.py` | SHAP waterfall plots (XML) |
| `generate_waterfall_combined_ext.py` | SHAP waterfall plots (JSON extended) |
| `generate_waterfall_combined_json.py` | SHAP waterfall plots (JSON) |
| `generate_waterfall_svm.py` | SHAP KernelExplainer pentru SVM |
| `fix_svm_barplot.py` | Fix vizualizare SHAP bar plot SVM |

### Aplicatie / Demo
| Script | Descriere |
|--------|-----------|
| `nutrition_recommender.py` | Modul recomandari nutritionale (importat de demo + streamlit) |
| `demo_end_to_end_v2.py` | Demo CLI: date → clasificare → recomandare + SHAP |
| `streamlit_app_v2.py` | Interfata web Streamlit |

## Structura date

```
data/
├── oura_combined.csv          # Date Oura procesate (1007 zile)
├── oura_historical.csv        # Export manual Oura (pre-API)
├── dataset_combinat*.csv      # Seturi integrate Oura + AW (per pipeline)
├── apple_health/              # CSV-uri din Apple Health export XML
├── Oura/raw/                  # CSV-uri brute din Oura API v2
└── Oura/Subscriptions/        # Facturi Oura (ignora)

apple_watch_api/
├── apple_health_daily.csv     # Date AW procesate (din JSON)
├── apple_health_auto.csv      # Date AW intermediare
└── raw_json/                  # JSON-uri brute Health Auto Export
```

## Rezultate (outputs/)

```
outputs/
├── figures/
│   ├── xml_pipeline/              # Figuri din pipeline XML (16 features)
│   ├── json_pipeline/             # Figuri din pipeline JSON (16 features)
│   ├── json_extended_pipeline/    # Figuri din pipeline JSON ext (19 features)
│   └── experiments/               # Figuri ablation, sensitivity, confidence, SMOTE
├── tables/                        # CSV-uri cu rezultate experimente
└── reports/                       # Rapoarte text comparatii surse
```

## Rezultate cheie (dupa fix rhr_avg, 30.03.2026)

- **Best model:** Random Forest, F1 Macro = 0.506
- **Best threshold:** T3 (Rest: readiness<60 OR sleep<60; Active: readiness>=85 AND sleep>=75)
- **Best imbalance strategy:** SMOTE k=3 (F1 Macro = 0.482)
- **Feature #1 cross-model:** aw_hr_mean
- **Ablation:** Combined +10.2pp F1 vs Oura-only
- **Dataset:** 1007 zile, 19 features, 3 clase (Rest/Normal/Active)
