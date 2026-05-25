# Gurbani Captioning

This is a prototype for Kirtan captioning. Please read more at this [blog post](https://karanbirsingh.com/gurbani-captioning/).

To integrate this into apps like Sikhi To The Max, much improvement is needed. Until then, this app can be used to gather feedback and serve as one example for others to improve on.

If you have other Sikhi projects or workflows and would like to chat, please feel free to [get in touch](https://www.karanbirsingh.com/).

## How to try it

You can [download the desktop apps](https://github.com/karanbirsingh/kirtan-captioning/releases).

If you want a quick preview, you can also check this [running website](https://bani.karanbirsingh.com).

## How well does it work?

We have authored a ['benchmark' here](https://karanbirsingh.github.io/live-gurbani-captioning-benchmark-v1/). This is a way to visualize and measure whether a system is doing the right thing.

The benchmark has four different 'modes':
1. The system must self-identify the Shabad and also follow along live
2. The system must self-identify the Shabad, but it is not live (for example, adding captions to a full recording)
3. The system is given the Shabad by user, but still needs to follow along live
4. The system is given the Shabad by user, and it is not live

One reason to share this code is to provide a baseline result that other folks can improve on. You can [see more here](https://github.com/karanbirsingh/kirtan-captioning/blob/main/benchmark/README.md) where this codebase has results for the harder mode #1.

## How it works

Overall, the system works like this:
1. Recent audio is provided to the system
2. ASR: The latest audio snippet is transcribed to Gurmukhi using an ASR model
3. Matcher: The transcription may have typos which is not OK for Gurbani, so it is matched to the closest Shabad line.
4. State maching: A 'state machine' keeps track of where we are and decides when to switch to a new line based on ongoing matcher updates

Each layer is independent. For example, the ASR model in this app is a small fine-tuned model that can run locally on most computers. You can swap it for a paid speech-to-text model from Google that will have higher accuracy.

## Quick Start

```bash
git clone https://github.com/karanbirsingh/kirtan-captioning.git
cd kirtan-captioning
pip install -r requirements.txt

# Auto-downloads the ONNX model on first run (~180 MB).
python server.py --desktop      # localhost + /mic/ UI
python server.py                # API mode, 0.0.0.0:8765
```

Open <http://127.0.0.1:8765/mic/>, click Start Mic, point it at kirtan audio.

## WebSocket Protocol

```
ws://localhost:8765/ws?cid=<client_id>
  Client → Server:  binary PCM (16kHz mono float32)
  Client → Server:  JSON commands  → see engine/wire.py
  Server → Client:  JSON events    → see engine/wire.py
```

`engine/wire.py` is the single source of truth for event/command schemas + `PROTOCOL_VERSION`. The `connected` event echoes the server's version; clients warn on mismatch.

## Tests

```bash
python tests/regression_sttm.py --check tests/regression_baseline.json  # matcher+SM
python tests/integration_server.py                                       # full WS stack
```

Both run in <5s with no audio / no real ONNX.

## License

CC BY-NC-SA 4.0 — see [LICENSE](LICENSE).
