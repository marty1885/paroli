var ws;
var ttsSocketURL = (window.location.toString().replace('http', 'ws') + 'api/v1/stream');
var pcmplayer_opt = {
     encoding: '16bitInt',
     channels: 1,
     sampleRate: 24000,
     flushingTime: 100
}
var player;

window.onload = function () {
    let speaker_list = document.getElementById('speaker_select')
    fetch('/api/v1/speakers').then(function (response) {
        return response.json();
    }).then(function (data) {
        let default_speaker = null
        for(let key in data) {
            let option = document.createElement('option')
            option.value = data[key]
            option.text = key
            speaker_list.appendChild(option)

            if(data[key] == 0) {
                default_speaker = data[key]
            }
        }
        speaker_list.value = default_speaker
    });
}

function runTTS() {
    let str = document.getElementById('txt_input').value
    let speaker = document.getElementById('speaker_select').value
    let data = JSON.stringify({text: str, speaker_id: parseInt(speaker)})
    if (ws == null || ws.readyState != 1) {
        player = new PCMPlayer(pcmplayer_opt);
        reconnectWS(function () {
            ws.send(data);
        });
        return;
    }

    ws.send(data);
}

function reconnectWS(connect_fn) {
     if (ws) ws.close()

     ws = new WebSocket(ttsSocketURL);
     ws.binaryType = 'arraybuffer';
     ws.addEventListener('open', function (event) {
          connect_fn && connect_fn()
     });
     ws.addEventListener('message', function (event) {
          var data = new Uint8Array(event.data);
          if(data.length == 0) return; // ignore empty (pong) messages
          player.feed(data);
     });
}
