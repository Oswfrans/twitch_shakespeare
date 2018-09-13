var tmi = require("tmi.js")
var fs = require('fs');


var options = {
    options: {
        debug: false
    },
    connection: {
        reconnect: true
    },
    identity: {
        username: "oswfrans",
        password: "oauth:dp2cebqa789olor6cykdzadgmd0yty"
    },
    channels: ["nalcs2",
    "tsm_dyrus",
    "sjow",
    "hotform",
    "nl_kripp",
    "scarra",
    "wingsofdeath",
    "tsm_theoddone",
    "trumpsc",
    "nvidia",
    "loltyler1",
    "itshafu",
    "frodan",
    "firebat",
    "iwilldominate",
    "meteos",
    "anniebot",
    "imaqtpie",
    "aphromoo",
    "riotgames"]
};

var client = new tmi.client(options);

client.connect();
var counter = 0;

var jsonfile = require('jsonfile')
var file = 'data.json'


var totalstring = ""

client.on("chat", function (channel, userstate, message, self) {
    // Don't listen to my own messages..
    if (self) return;
    message = message.replace(/[^\w\s]/gi, '')
    message = message + "<eos>"
    totalstring += message
    console.log(message);

    counter++;
    if(counter % 200 == 0)
    {
    	fs.writeFile("database2.txt", totalstring, function(err) {});
    }
});
