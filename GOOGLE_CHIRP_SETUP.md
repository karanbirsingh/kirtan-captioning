# Google Chirp 2 setup

Bani Mic's default ASR runs on-device for free. **Google Chirp 2** is an
opt-in cloud backend that bumps accuracy on hard audio, at **~$3/hour**
of listening time (billed by Google to your own Google Cloud account).
The mic auto-stops after 1 hour of Chirp 2 being active as a cost rail.

You'll do this once, takes ~5 minutes. There's a CLI version at the
bottom if you prefer the terminal.

## Setup (GCP Console, 5 steps)

### 1. Pick a project and link billing

Go to <https://console.cloud.google.com/projectcreate> and create a
project (e.g. `bani-mic`), or pick one you already have. Make sure
**billing is enabled** on it — link a billing account from
<https://console.cloud.google.com/billing>.

### 2. Enable the Speech-to-Text API

Open <https://console.cloud.google.com/apis/library/speech.googleapis.com>
and click **Enable**.

### 3. Create a service account

Open <https://console.cloud.google.com/iam-admin/serviceaccounts/create>
and:

1. Name it `bani-mic` → **Create and continue**
2. Grant role **Cloud Speech Client** → **Continue**
3. **Done**

### 4. Download a key

In the service account list, click `bani-mic@…` → **Keys** tab →
**Add Key → Create new key → JSON → Create**. Your browser downloads
a JSON file.

### 5. Upload it to Bani Mic

In the app: **Settings → ASR Engine → Google Chirp 2 → Choose key
file…** → pick the JSON you just downloaded. The pill in the status
bar turns amber "Google Chirp" once it's active.

> Wait ~30 seconds after granting the role in step 3 before testing.
> Google's IAM takes that long to propagate; transcribe calls in the
> first 30 seconds can come back as "permission denied" then start
> working on their own.

## Set a billing cap

Strongly recommended — protects against bugs, mistakes, or anyone
gaining access to the key file:

<https://console.cloud.google.com/billing/budgets> → **Create budget**
→ set a monthly cap (e.g. $50) with email alert at 50% / 90% / 100%.

## CLI version

For users with `gcloud` already set up — same flow in 4 commands.
Replace `<PROJECT>` with your project ID:

```bash
gcloud services enable speech.googleapis.com --project=<PROJECT>

gcloud iam service-accounts create bani-mic --project=<PROJECT>

gcloud projects add-iam-policy-binding <PROJECT> \
  --member="serviceAccount:bani-mic@<PROJECT>.iam.gserviceaccount.com" \
  --role="roles/speech.client"

gcloud iam service-accounts keys create ~/bani-mic-key.json \
  --iam-account="bani-mic@<PROJECT>.iam.gserviceaccount.com"
```

Then upload `~/bani-mic-key.json` via Settings as in step 5.

## Where the key lives

- **Windows**: `%LOCALAPPDATA%\bani-mic\google_credentials.json`
- **macOS**: `~/Library/Application Support/bani-mic/google_credentials.json`
- **Linux**: `$XDG_DATA_HOME/bani-mic/google_credentials.json`

To remove it: **Settings → Disconnect & remove key**, or delete the
file directly.

To invalidate it server-side as well (recommended if the key was ever
exposed): go to the service account's Keys tab in GCP Console and
delete it there.
