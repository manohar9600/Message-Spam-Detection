var amqp = require('amqplib/callback_api')


// functions
var msg

function detect_spam() {
    msg = document.getElementById('sms').value
    //msg = 'Hello Manu'
    send_msg(msg)
    console.log("End")
}

function write_html(res){
    document.getElementById('output').innerHTML = res
    return
}


// Network Code
function send_msg(msg) {
    amqp.connect('amqp://localhost', function (err, conn) {
        conn.createChannel(function (err, ch) {
            ch.assertQueue('', {
                exclusive: true
            }, function (err, q) {
                var corr = generateUuid();

                console.log(' [x] Requesting detect_spam(%s)', msg);

                ch.consume(q.queue, function (msg) {
                    if (msg.properties.correlationId == corr) {
                        console.log(' [.] Got %s', msg.content.toString());
                        write_html(msg.content.toString());
                        setTimeout(function () {
                            conn.close();
                            return;
                            },
                            500);    
                    }
                }, {
                    noAck: false
                });
                ch.sendToQueue('rpc_queue', new Buffer(msg), {
                    correlationId: corr,
                    replyTo: q.queue
                });

            });
        });
    });
}

function generateUuid() {
    return Math.random().toString() +
        Math.random().toString() +
        Math.random().toString();
}