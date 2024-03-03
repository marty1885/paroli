# paroli-server API

## REST APIs

### /api/v1/speakers

* Method: GET
* Parameters: None

Returns a mapping from speaker name to speaker ID

```json
{
	"Arlan": 13,
	"Asta": 8,
	"Bailu": 10,
	"Banxia": 21,
	"Herta": 18,
	"Qingni": 16,
	"Qingque": 22,
	"Qingzu": 23,
	"Sushang": 19,
	"Yanqing": 14,
	"bronya": 6,
	"clara": 2,
	"danshu": 17,
	"fuxuan": 12,
	"himeko": 7,
	"hook": 4,
	"march7th": 0,
	"natasha": 1,
	"pela": 9,
	"pom": 15,
	"seele": 5,
	"serval": 3,
	"silverwolf": 20,
	"tingyun": 11
}
```

### /api/v1/synthesise

* Method: POST
* Parameters: A JSON object denoting the text and (optional) speaker
* Response: `audio/ogg; codecs=opus` or `audio/raw`

example request body:
```json
{
    "text": "How can I help you? Is there anything wrong?",
    "speaker_id": 8
}
```

format of the request JSON

```c++
struct ApiData
{
    std::string text;
    std::optional<uint64_t> speaker_id;
    std::optional<float> length_scale;
    std::optional<float> noise_scale;
    std::optional<float> noise_w;
    // The returned audio format. Vaild values are "pcm" and "opus"
    // If none given, the default is "opus". Under pcm mode, the
    // sample rate is whatever the loaded model provides. Under opus
    // sample rate is always 24K
    std::optional<std::string> audio_formt;
};

```

example response:

```
<Some OGG/OPUS audio>
```

## WebSocket API

### /api/v1/stream

* Method: GET
* Parameters: None

This endpoint works exactly like the synthesise API above. But audio is streamed in chunk as soon as it can - reducing latency, as binary blobs. Message format is the same as the synthesise API body. A text message is sent once an error is encountered or synthesis of current text is finished.

For example, the following message causes OPUS audio to be streamed back as binray messages.

```bash
wscat -c 'ws://example.com:8848/api/v1/stream' 
> {"text": "Hello! how can I help you"}
< [OPUS audio blob]
< [OPUS audio blob]
< {"status":"ok", "message":"finished"}
```

The server will reply error as text

```bash
wscat -c 'ws://example.com:8848/api/v1/stream' 
> {"hello": "blablabla"}
< {"status":"failed", "message":"Missing 'text' field"}
```
