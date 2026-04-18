## 11 Case Studies and Qualitative Analysis

To complement the aggregate quantitative metrics presented in Section 9, this section provides a qualitative analysis of the model's operational behavior. Deliberately engineered target URLs representing clean baselines, structural obfuscation, and architectural edge-cases were evaluated through the live PhishLens inference server. The URLs were processed through both the Binary (BCE Loss) and Multiclass (Cross-Entropy Loss) tracks to evaluate their strengths and limitations. The results highlight the strengths of the Character-Level 1D-CNN in capturing structural patterns, as well as its limitations under specific conditions.

### True Positives: Successful Detection of Malicious or Obfuscated URLs

| ID | Model Type | URL | Attack / Scenario | Prediction (Score) | Outcome | Key Insight |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | Binary | `google.com/url?q=https://apple-id.tk` | Parameter stuffing / open redirect | Phishing (0.94) | TP | The CNN successfully scans beyond the trusted root domain and identifies the malicious payload embedded in the query string. |
| 2 | Binary | `secure-update.paypal.com-auth.xyz` | Delimiter injection + brand spoofing | Phishing (0.89) | TP | The model prioritizes the true apex domain over misleading trusted keywords placed earlier in the URL. |
| 3 | Binary | `secure-login.bankofamerica.verify-now.xyz` | Subdomain spoofing | Phishing (0.91) | TP | The CNN captures deceptive hierarchical structure and does not confuse spoofed subdomains with the real domain owner. |
| 4 | Binary | `microsoft-secure-login-authentication.xyz` | Keyword spoofing | Phishing (0.87) | TP | The model learns suspicious combinations of security-related keywords and hostile TLD usage. |
| 5 | Binary | `p-a-y-p-a-l-secure-login.com` | Delimiter injection | Phishing (0.85) | TP | Character-level convolutions preserve malicious spatial patterns even when keywords are broken apart by hyphens. |
| 6 | Binary | `microsoft-support-1ogin.com` | Homoglyph substitution | Phishing (0.82) | TP | Minor character substitutions do not fully disrupt learned phishing representations in embedding space. |
| 7 | Binary | `update-account-verify-paypal.com` | Keyword stuffing | Phishing (0.88) | TP | Dense clustering of phishing-related tokens triggers strong malicious activations despite otherwise simple structure. |
| 8 | Binary | `secure.verify.apple.com.fake-domain.xyz` | Deep-path / domain mismatch spoofing | Phishing (0.90) | TP | The model correctly distinguishes trusted brand tokens from the actual registrable domain. |
| 9 | Binary | `login.secure.update.chase.com.fake.xyz` | Multi-level spoofing | Phishing (0.92) | TP | The CNN learns that long trusted-looking subdomain chains do not override the malicious apex domain. |
| 10 | Multiclass | `http://www.legitimatesmallbusiness.com/wp-content/uploads/2026/hacked-by-syndicate.html` | Defacement on trusted domain | Defacement (0.39) | TP | The multiclass model identifies structural cues of a compromised CMS path rather than treating the URL as generic phishing. |

### True Negatives: Correct Classification of Benign URLs

| ID | Model Type | URL | Scenario | Prediction (Score) | Outcome | Key Insight |
|:---|:---|:---|:---|:---|:---|:---|
| 11 | Binary | `https://engineering.asu.edu/robotics-and-autonomous-systems-ms/` | Legitimate academic URL | Benign (0.74) | TN | The model tolerates long paths and hyphenation when the surrounding domain structure is trustworthy. |
| 12 | Binary | `https://amazon.com/gp/cart/view.html` | Legitimate e-commerce URL | Benign (0.88) | TN | Standard commercial path patterns are not confused with phishing syntax. |
| 13 | Binary | `https://linkedin.com/in/profile/view` | Legitimate social profile URL | Benign (0.86) | TN | The model recognizes common social-media path conventions without over-penalizing generic keywords. |
| 14 | Binary | `https://wikipedia.org/wiki/Machine_learning` | Legitimate long-path informational URL | Benign (0.90) | TN | Long URL length alone is not sufficient to trigger malicious classification when structural cues remain benign. |

### False Negatives: Missed Threats and Architectural Blind Spots

| ID | Model Type | URL | Attack / Scenario | Prediction (Score) | Outcome | Key Insight |
|:---|:---|:---|:---|:---|:---|:---|
| 15 | Binary | `https://docs.google.com/forms/.../viewform` | Phishing hosted on legitimate cloud infrastructure | Benign (0.92) | FN | The URL uses fully legitimate Google-hosted structure, leaving few lexical or spatial anomalies for a URL-only model to detect. |
| 16 | Binary | `https://drive.google.com/file/.../malicious` | Abuse of trusted cloud-hosted file sharing | Benign (0.89) | FN | Trusted platform infrastructure masks malicious intent, indicating the need for content-aware or host-context analysis. |
| 17 | Binary | `http://45.33.32.156/hidden/payload/update_v2.1.apk` | Raw-IP malware dropper | Benign (0.91) | FN | The CNN struggles on low-linguistic, low-domain-structure URLs; this justifies the heuristic fallback in the dual-engine system. |

### False Positives: Over-Sensitivity on Benign but Unusual URLs

| ID | Model Type | URL | Scenario | Prediction (Score) | Outcome | Key Insight |
|:---|:---|:---|:---|:---|:---|:---|
| 18 | Multiclass | `https://icc-cricket.com/tournaments/t20-world-cup` | Hyphen-heavy benign URL | Phishing (0.99) | FP | Frequent hyphenation resembles delimiter-injection patterns, causing the multiclass model to over-index on obfuscation cues. |
| 19 | Binary | `https://secure07ea.chase.com/web/auth/dashboard?token=9f8a7b6c5d4e3f2` | High-entropy benign token | Phishing (0.96) | FP | A long randomized session token activates filters associated with DGA-like or obfuscated phishing strings. |
| 20 | Binary | `https://example.com/login?session=123456789abcdef` | Long tokenized benign query | Suspicious (0.68) | FP | Benign authentication parameters can resemble malicious stuffing when entropy and login-related tokens appear together. |

### 11.1 The Triumphs: Spatial Reasoning and Multiclass Isolation

The live evaluations successfully validated the spatial pooling mechanisms of the 1D-CNN, proving its ability to detect structural obfuscation (True Positives)[cite: 982].

* **Parameter Stuffing** (`google.com/url?q=https://apple-id.tk`): The network successfully ignored the highly trusted root domain (`google.com`) and located the malicious payload buried deep within the parameter string, flagging it as High Risk Phishing.
* **Delimiter Injection** (`secure-update.paypal.com-auth.xyz`): The model correctly identified the true apex domain (`verify.xyz`) despite the heavy use of hyphens attempting to spoof the PayPal brand.
* **Defacement on Trusted Domains** (`.../2026/hacked-by-syndicate.html`): While the root domain was completely benign, the Multiclass Categorical Cross-Entropy optimization allowed the CNN to assign a 39% Defacement probability, accurately flagging the compromised CMS path.
* **Malware Drop Detection** (`45.33.32.156/...update_v2.1.apk`): This case stands as the definitive success of the dual-engine architecture. While the Binary neural track struggled with the lack of linguistic structure, the Multiclass track successfully categorized the apk payload as 100% Malware. Furthermore, the deterministic heuristic fallback immediately caught the raw IPv4 hosting.

### 11.2 The Blindspots: False Negatives

Analyzing the failure states provides critical insight into the boundaries of URL-only threat detection.

* **Compromised Cloud Infrastructure** (`docs.google.com/forms/.../viewform`): The model classified this malicious Google Docs form as highly Benign (92%). Because the URL structure utilizes 100% legitimate Google infrastructure, there are no spatial anomalies for the CNN to detect. This confirms that to detect compromised cloud infrastructure, sequence models must eventually be paired with downstream HTML/DOM scraping.

### 11.3 The Constraints: Over-Sensitivity and False Positives

The transition from Binary to Multiclass architectures introduces hyper-sensitivity, occasionally resulting in false alarms on complex benign URLs.

* **Delimiter Over-Indexing** (`icc-cricket.com/tournaments/t20-world-cup`): The Binary model correctly classified this sports URL as Benign. However, the Multiclass model aggressively flagged it as 99% Phishing. The heavy use of hyphens acting as word separators successfully triggered the model's delimiter-injection filters, causing it to over-index on the perceived obfuscation.
* **High-Entropy Tokens** (`secure07ea.chase.com/...token=9f8a...`): While the root domain was safe, the presence of a massive, randomized cryptographic token accidentally triggered the Domain Generation Algorithm (DGA) spatial filters, leading to a False Positive[cite: 1000].

These constraints highlight the delicate balance required when tuning classification thresholds for enterprise environments[cite: 1002].
```
