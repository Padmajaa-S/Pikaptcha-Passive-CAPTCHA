

# PIKAPTCHA - Passive Captcha Solution

**PIKAPTCHA** is a 2-layer machine learning-based CAPTCHA solution designed for seamless bot detection and prevention against DOS and DDOS attacks. Unlike traditional CAPTCHAs, PIKAPTCHA uses passive data collection and minimal user interaction to identify and block bot activities while ensuring a smooth user experience. 
![FINAL](https://github.com/user-attachments/assets/40d90625-0141-494a-830e-11d5add592b6)

## Features
- **Behavioral Analysis**: Uses typing patterns, mouse movement trajectories, pauses, touch sensitivity, scrolling patterns, and device fingerprinting to detect bot-like behavior.
- **Honeypots & Trap Links**: Hidden interactive elements trigger alerts if bots interact, without affecting user experience.
- **Adaptive Challenge**: When suspected bot behavior is detected, a final, randomized adversarial network image challenge is presented (e.g., "Connect the Dots," simple true/false math).
- **Request Timeouts & Throttling**: Automatically manages server requests, protecting against DDOS overload.

## Unique Value Points
- **Non-intrusive**: Users don’t need to solve traditional CAPTCHAs. Instead, the solution is passively trained to recognize bot behavior.
- **Highly Accurate**: The model, using SVM and LSTM classifiers, achieves a 97-98% accuracy rate for bot detection.
- **API Pluggable**: Can be easily integrated into various applications with minimal configuration.
- **Bot Tracking**: PIKAPTCHA also provides insights into bot tracking by monitoring communications and analyzing C2 protocols.

## How It Works
1. **Data Collection**: Passive data (e.g., typing speed, mouse movements) is collected without interrupting the user.
2. **Behavioral Model Training**: Data is fed into a 3-layer machine learning model (using SVM and LSTM classifiers) to distinguish human users from bots.
3. **Real-Time Detection**: Based on user behavior, the model assesses the likelihood of bot activity in real time.
4. **Final Challenge (If Needed)**: A final overlaid visual CAPTCHA is presented to confirm human presence if bot behavior is suspected.
5. **Bot Identification & Tracking**: PIKAPTCHA goes beyond detection to track bots through communications, C2 protocol analysis, domain, and IP analysis.

## Security Benefits
- **Reduces Abandonment**: Unlike traditional CAPTCHAs, PIKAPTCHA ensures a faster, frustration-free user experience.
- **Hard to Circumvent**: No keywords or predictable patterns, and code compression minimizes reverse-engineering potential.
- **Collaborative Tracking**: Works with ISPs or law enforcement for in-depth bot tracking if needed.

## Usage
1. Integrate PIKAPTCHA's API in your web application or service as an additional security layer.
2. The model automatically captures user behavior and begins assessing for bot-like activity.
3. If suspicious behavior is detected, the passive CAPTCHA can prompt the user with an interactive challenge, verifying their humanity without disrupting the experience.


## Contributing
We welcome contributions! Please fork the repository, make changes, and submit a pull request. Be sure to include tests for new features or updates.

