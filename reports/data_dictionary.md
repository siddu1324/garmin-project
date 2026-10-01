# Data dictionary

Every CSV cell is read as text with pandas default missing-value coercion disabled.
Original source cells are never rewritten. Blank/whitespace and literal NA/N/A/NaN/nan/NULL/null
are candidate missing markers in structured fields; free-text fields retain nonblank tokens as text.
-99/-999 are explicitly screened as possible sentinels, not
asserted to be valid coding for every questionnaire. The JSON profiles count observed blanks,
candidate missing cells and numeric/integer cells by field in development participants only.
Zeros remain values; absence of a sensor record is separate from a blank cell.

Garmin sensor categories and provisional units:

| data_type | Meaning | Provisional units | Initial role |
| --- | --- | --- | --- |
| ibi | Beat interval stream, not a raw ECG/PPG waveform | milliseconds | HRV after timing/quality validation |
| hr | Heart rate | beats/minute | candidate HR summaries |
| pulseOx | Blood oxygen saturation | percent | candidate, conditional on usable coverage |
| stress | Proprietary Garmin stress | score approximately 0–100 | exploratory only, excluded |
| steps | Predominantly cumulative daily steps | steps | validated increments later |
| calories | Predominantly cumulative daily energy | likely kcal; exporter unverified | validated increments later |
| floorsClimbed | Floor counter | floors | optional after reset validation |
| intensityMinutes | Intensity counter | minutes; weighting unverified | optional after reset validation |

No respiratory waveform or raw ECG/PPG stream is present. Embedded EMA device fields have no
verified preceding-measurement anchor and are excluded from the initial wearable predictors.
Free text and location are sensitive and excluded. Unknown survey codes are not inferred from
field names. Morning/Midday/Endday categories are identified by schema, not participant filenames.

All source fields follow. Field names are public schema metadata; no real values or IDs appear.

## Morning

| field | semantic_category | interpretation | units | observed_development_type | development_missing_cells |
| --- | --- | --- | --- | --- | --- |
| mbl_cod | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| rsp_id | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| instance_id | clock | numeric epoch stored as text | Unix seconds; instance is scheduled | numeric | 0 |
| scheduled_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| timezone_offset | clock | integer stored as text | minutes local minus UTC | numeric | 0 |
| actual_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| HAPPY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| ANGRY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| CONFIDENT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| SAD_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| EXCITED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| GRATEFUL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| SCARED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| AMUSED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| CONTENT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| PROUD_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| WORRIED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| CALM_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| HOPEFUL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| TIRED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| INSPIRED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| RELAXED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| GUILTY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| SLEEP_SLEEPTIME | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | text/mixed | 3 |
| SLEEP_LATENCY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| SLEEP_WAKETIME | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | text/mixed | 3 |
| SLEEP_QUALITY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| PSS_NERVOUS | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 3 |
| PSS_COPE | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 3 |
| PSS_CONTROL | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 3 |
| SE_PREG_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 769 |
| SE_WORK_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 801 |
| SE_MONEY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 872 |
| SE_PARTNER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 868 |
| SE_FAMILY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 901 |
| SE_FRIEND_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 928 |
| SE_ARGUE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 915 |
| SE_RACISM_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_OTHDISC_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_THREAT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_LEGAL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 930 |
| SE_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 846 |
| SE_NONE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 392 |
| SE_RACISM_DIRECT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_RACISM_VICAR_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_RACISM_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_GENDER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_AGE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_WEIGHT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_EDUC_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_CLASS_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_DISABIL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| THREAT_MESSAGE_1 | sensitive survey text | text/coded content; values withheld | not a predictor | empty | 934 |
| SE_OTHER_TEXT_1 | sensitive survey text | text/coded content; values withheld | not a predictor | text/mixed | 852 |
| SE_MOST_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 536 |
| SE_TIMING_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 536 |
| SE_RESOLVED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 536 |
| SE_STRESSFUL_RATE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 537 |
| SE_CONTROL_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 537 |
| SE_IMPACT_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 538 |
| SE_NONE_WHY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 396 |
| SE_NONE_EXPECT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 396 |
| SE_NONE_EXPPOS_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 396 |
| PE_SOCIAL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 5 |
| SOCIAL_PARTNER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 415 |
| SOCIAL_PARENT_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 767 |
| SOCIAL_FAMILY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 772 |
| SOCIAL_FRIEND_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 792 |
| SOCIAL_COWORKER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 816 |
| SOCIAL_COMMUNITY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 912 |
| SOCIAL_DOCTOR_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 925 |
| SOCIAL_STRANGER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 908 |
| SOCIAL_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 914 |
| SUPPORT_EMO_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 724 |
| SUPPORT_INFO_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 836 |
| SUPPORT_INST_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 814 |
| SUPPORT_PROV_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 813 |
| SUPPORT_DISC_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 619 |
| SUPPORT_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 753 |
| SOCIAL_MODE_F2F_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 339 |
| SOCIAL_MODE_PHONE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 666 |
| SOCIAL_MODE_SM_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 880 |
| SOCIAL_MODE_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 928 |
| NOSOCIAL_ALONE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 683 |
| NOSOCIAL_ACTIVITY_1_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 896 |
| NOSOCIAL_ACTIVITY_2_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 864 |
| NOSOCIAL_ACTIVITY_3_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 910 |
| NOSOCIAL_ACTIVITY_4_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 911 |
| NOSOCIAL_ACTIVITY_5_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 928 |
| NOSOCIAL_ACTIVITY_6_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 930 |
| NOSOCIAL_ACTIVITY_7_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 918 |
| NOSOCIAL_ACTIVITY_8_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 926 |
| NOSOCIAL_ACTIVITY_9_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 921 |
| NOSOCIAL_ACTIVITY_10_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 929 |
| NOSOCIAL_ACTIVITY_11_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 826 |
| PE_PLEASURE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 769 |
| PE_MASTERY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 669 |
| PE_RELAX_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 605 |
| PE_SPIRITUAL_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 850 |
| PE_UNEXPECTED_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 886 |
| PE_OTHER_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 850 |
| PE_NONE_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 627 |
| MORNING_SURVEY_COMPLETED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| STRESS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| PULSE_OX | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| HEART_RATE | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| RESPIRATION | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| BODY_BATTERY | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| STEPS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| CALORIES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| FLOORS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| INTENSITY_MINUTES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| LATITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| LONGITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| Location Timestamp | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 376 |
| Location near window | sensitive location | source text | not a predictor; exporter semantics unverified | text/mixed | 0 |
| Proximity to window | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 376 |
| AVG_AMP | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| VOX_ACTV | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |

## Midday

| field | semantic_category | interpretation | units | observed_development_type | development_missing_cells |
| --- | --- | --- | --- | --- | --- |
| mbl_cod | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| rsp_id | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| instance_id | clock | numeric epoch stored as text | Unix seconds; instance is scheduled | numeric | 0 |
| scheduled_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| timezone_offset | clock | integer stored as text | minutes local minus UTC | numeric | 0 |
| actual_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| HAPPY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| ANGRY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| CONFIDENT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| SAD | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1 |
| EXCITED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| GRATEFUL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| SCARED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| AMUSED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 3 |
| CONTENT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| PROUD | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| WORRIED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| CALM | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| HOPEFUL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| TIRED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| INSPIRED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| RELAXED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| GUILTY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PSS_NERVOUS | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 11 |
| PSS_COPE | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 12 |
| PSS_CONTROL | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 12 |
| SE_PREG | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1560 |
| SE_WORK | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1492 |
| SE_MONEY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1720 |
| SE_PARTNER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1752 |
| SE_FAMILY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1746 |
| SE_FRIEND | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1808 |
| SE_ARGUE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1786 |
| SE_RACISM | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1834 |
| SE_OTHDISC | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1833 |
| SE_THREAT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1835 |
| SE_LEGAL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1834 |
| SE_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1678 |
| SE_NONE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 812 |
| SE_RACISM_DIRECT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1837 |
| SE_RACISM_VICAR | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1836 |
| SE_RACISM_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_GENDER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_AGE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_WEIGHT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_EDUC | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_CLASS | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 1838 |
| SE_OTHDISC_DISABIL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1836 |
| SE_OTHDISC_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1836 |
| THREAT_MESSAGE | sensitive survey text | text/coded content; values withheld | not a predictor | numeric | 1836 |
| SE_OTHER_TEXT | sensitive survey text | text/coded content; values withheld | not a predictor | text/mixed | 1684 |
| SE_MOST | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1036 |
| SE_TIMING | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1036 |
| SE_RESOLVED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1037 |
| SE_STRESSFUL_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1039 |
| SE_CONTROL_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1039 |
| SE_IMPACT_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1039 |
| SE_NONE_WHY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 812 |
| SE_NONE_EXPECT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 813 |
| SE_NONE_EXPPOS | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 813 |
| PE_SOCIAL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 21 |
| SOCIAL_PARTNER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 841 |
| SOCIAL_PARENT | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1396 |
| SOCIAL_FAMILY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1356 |
| SOCIAL_FRIEND | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1389 |
| SOCIAL_COWORKER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1424 |
| SOCIAL_COMMUNITY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1736 |
| SOCIAL_DOCTOR | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1793 |
| SOCIAL_STRANGER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1734 |
| SOCIAL_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1765 |
| SUPPORT_EMO | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1309 |
| SUPPORT_INFO | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1541 |
| SUPPORT_INST | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1486 |
| SUPPORT_PROV | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1450 |
| SUPPORT_DISC | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1095 |
| SUPPORT_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1462 |
| SOCIAL_MODE_F2F | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 518 |
| SOCIAL_MODE_PHONE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1051 |
| SOCIAL_MODE_SM | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1716 |
| SOCIAL_MODE_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1831 |
| NOSOCIAL_ALONE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1564 |
| NOSOCIAL_ACTIVITY_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1768 |
| NOSOCIAL_ACTIVITY_2 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1777 |
| NOSOCIAL_ACTIVITY_3 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1788 |
| NOSOCIAL_ACTIVITY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1797 |
| NOSOCIAL_ACTIVITY_5 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1831 |
| NOSOCIAL_ACTIVITY_6 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1827 |
| NOSOCIAL_ACTIVITY_7 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1816 |
| NOSOCIAL_ACTIVITY_8 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1819 |
| NOSOCIAL_ACTIVITY_9 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1821 |
| NOSOCIAL_ACTIVITY_10 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1831 |
| NOSOCIAL_ACTIVITY_11 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1757 |
| PE_PLEASURE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1242 |
| PE_MASTERY | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 971 |
| PE_RELAX | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1240 |
| PE_SPIRITUAL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1716 |
| PE_UNEXPECTED | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1700 |
| PE_OTHER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1644 |
| PE_NONE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 1507 |
| STRESS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| PULSE_OX | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| HEART_RATE | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| RESPIRATION | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| BODY_BATTERY | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| STEPS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| CALORIES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| FLOORS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| INTENSITY_MINUTES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| LATITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| LONGITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| Location Timestamp | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 750 |
| Location near window | sensitive location | source text | not a predictor; exporter semantics unverified | text/mixed | 0 |
| Proximity to window | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 750 |
| AVG_AMP | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| VOX_ACTV | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |

## Endday

| field | semantic_category | interpretation | units | observed_development_type | development_missing_cells |
| --- | --- | --- | --- | --- | --- |
| mbl_cod | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| rsp_id | identity | opaque string; never numeric | none; exact keys only | numeric | 0 |
| instance_id | clock | numeric epoch stored as text | Unix seconds; instance is scheduled | numeric | 0 |
| scheduled_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| timezone_offset | clock | integer stored as text | minutes local minus UTC | numeric | 0 |
| actual_start_local | clock | naive local datetime text | local clock; offset required | text/mixed | 0 |
| HAPPY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| ANGRY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| CONFIDENT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| SAD_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| EXCITED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| GRATEFUL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| SCARED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| AMUSED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| CONTENT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| PROUD_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| WORRIED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| CALM_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| HOPEFUL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| TIRED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| INSPIRED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| RELAXED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| GUILTY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 0 |
| PSS_NERVOUS | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 0 |
| PSS_COPE | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 1 |
| PSS_CONTROL | outcome item | integer code | 1–5 codes; displayed 0–4 labels | numeric | 2 |
| SE_PREG_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 779 |
| SE_WORK_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 844 |
| SE_MONEY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 876 |
| SE_PARTNER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 875 |
| SE_FAMILY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 888 |
| SE_FRIEND_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 919 |
| SE_ARGUE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 898 |
| SE_RACISM_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 932 |
| SE_OTHDISC_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 930 |
| SE_THREAT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 931 |
| SE_LEGAL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 860 |
| SE_NONE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 379 |
| SE_RACISM_DIRECT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_RACISM_VICAR_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_RACISM_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| SE_OTHDISC_GENDER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_AGE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_WEIGHT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_EDUC_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_CLASS_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | empty | 934 |
| SE_OTHDISC_DISABIL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 932 |
| SE_OTHDISC_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 933 |
| THREAT_MESSAGE_4 | sensitive survey text | text/coded content; values withheld | not a predictor | numeric | 931 |
| SE_OTHER_TEXT_4 | sensitive survey text | text/coded content; values withheld | not a predictor | text/mixed | 865 |
| SE_MOST_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_TIMING_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_RESOLVED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_STRESSFUL_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_CONTROL_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_IMPACT_RATE | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 559 |
| SE_NONE_WHY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 371 |
| SE_NONE_EXPECT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 371 |
| SE_NONE_EXPPOS_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 371 |
| PE_SOCIAL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 2 |
| SOCIAL_PARTNER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 278 |
| SOCIAL_PARENT_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 646 |
| SOCIAL_FAMILY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 630 |
| SOCIAL_FRIEND_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 668 |
| SOCIAL_COWORKER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 849 |
| SOCIAL_COMMUNITY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 896 |
| SOCIAL_DOCTOR_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 923 |
| SOCIAL_STRANGER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 890 |
| SOCIAL_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 911 |
| SUPPORT_EMO_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 627 |
| SUPPORT_INFO_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 783 |
| SUPPORT_INST_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 734 |
| SUPPORT_PROV_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 744 |
| SUPPORT_DISC_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 483 |
| SUPPORT_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 763 |
| SOCIAL_MODE_F2F_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 202 |
| SOCIAL_MODE_PHONE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 527 |
| SOCIAL_MODE_SM_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 858 |
| SOCIAL_MODE_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 926 |
| NOSOCIAL_ALONE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 820 |
| NOSOCIAL_ACTIVITY_1_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 920 |
| NOSOCIAL_ACTIVITY_2_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 909 |
| NOSOCIAL_ACTIVITY_3_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 899 |
| NOSOCIAL_ACTIVITY_4_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 916 |
| NOSOCIAL_ACTIVITY_5_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 931 |
| NOSOCIAL_ACTIVITY_6_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 932 |
| NOSOCIAL_ACTIVITY_7_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 920 |
| NOSOCIAL_ACTIVITY_8_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 929 |
| NOSOCIAL_ACTIVITY_9_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 931 |
| NOSOCIAL_ACTIVITY_10_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 932 |
| NOSOCIAL_ACTIVITY_11_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 902 |
| PE_PLEASURE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 509 |
| PE_MASTERY_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 540 |
| PE_RELAX_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 496 |
| PE_SPIRITUAL_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 863 |
| PE_UNEXPECTED_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 848 |
| PE_OTHER_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 804 |
| PE_NONE_4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 807 |
| ENDDAY_STRESS | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 4 |
| ENDDAY_TYPICAL | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 5 |
| ENDDAY_STRESS_PREG | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 5 |
| ENDDAY_STRESS_WORK | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 5 |
| ENDDAY_STRESS_HOME | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 5 |
| ENDDAY_STRESS_PARTNER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| ENDDAY_STRESS_FRIEND | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| GAC_1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| GAC_2 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| GAC_3 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| PES_HASSLES_Q1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| PES_HASSLES_Q2 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| PES_HASSLES_Q3 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 6 |
| PES_HASSLES_Q4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 7 |
| PES_HASSLES_Q5 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PES_UPLIFTS_Q1 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PES_UPLIFTS_Q2 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PES_UPLIFTS_Q3 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PES_UPLIFTS_Q4 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| PES_UPLIFTS_Q5 | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 8 |
| ENDDAY_TEXT | sensitive survey text | text/coded content; values withheld | not a predictor | text/mixed | 8 |
| ENDDAY_REMINDER | survey response | source text / numeric codes | questionnaire-specific; not an initial predictor | numeric | 9 |
| STRESS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| PULSE_OX | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| HEART_RATE | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| RESPIRATION | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| BODY_BATTERY | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| STEPS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| CALORIES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| FLOORS | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| INTENSITY_MINUTES | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| LATITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| LONGITUDE | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 0 |
| Location Timestamp | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 400 |
| Location near window | sensitive location | source text | not a predictor; exporter semantics unverified | text/mixed | 0 |
| Proximity to window | sensitive location | source text | not a predictor; exporter semantics unverified | numeric | 400 |
| AVG_AMP | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |
| VOX_ACTV | embedded EMA device/context field | source text | timing/units unverified; exclude initial predictors | numeric | 0 |

## Garmin

| field | semantic_category | interpretation | units | observed_development_type | development_missing_cells |
| --- | --- | --- | --- | --- | --- |
| ts | clock | numeric epoch stored as text | Unix seconds; acquisition-versus-receipt semantics unconfirmed | — | — |
| dte_tme | clock | naive local datetime text | local clock; offset required | — | — |
| rsp_id | identity | opaque string; never numeric | none; exact keys only | — | — |
| mbl_cod | identity | opaque string; never numeric | none; exact keys only | — | — |
| data_type | sensor category | categorical text | see sensor dictionary | — | — |
| val | sensor value | numeric stored as text | depends on data_type | — | — |
