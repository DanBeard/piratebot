const Quartermaster = (canvas, ws) => {

    // draw bg
    let bgRect = new fabric.Rect({
        originX: 'center',
        originY: 'top',
        width: 300,
        height: 700,
        fill: 'rgba(100,100,100,0.5)'
    });

    const fontStyle =  {
        fill: 'black',
        strokeWidth: 1,
        stroke: "black",
        textAlign: "center",
        //fontFamily: "\"Press Start 2P\", Monaco, monospace",
        fontSize: 18,
        objectCaching:false,
        selectable: false
    };
    canvas.add(bgRect);

    let title = new fabric.Text('QUARTERMASTER', {
        originX: "center",
        originY: "center",
        top: 20,
        ...fontStyle
    });

    let state_title = new fabric.Text('STATE:NULL', {
        originX: "center",
        originY: "center",
        top: 50,
        ...fontStyle
    });

    let play_song = new fabric.Text('Sing a Shanty [S]', {
        originX: "center",
        originY: "center",
        top: 150,
        ...fontStyle
    });

    let cannon = new fabric.Text('Fire cannon [Space]', {
        originX: "center",
        originY: "center",
        top: 200,
        ...fontStyle
    });

    document.addEventListener('keyup', event => {
        if (event.key === 's') {
            ws.send(JSON.stringify({
                type : "C",
                address: "PUMPKINS",
                data:{command:"SING"}
            }));
        }
    });


    const group = new fabric.Group([bgRect, title, state_title, play_song, cannon], {
        top:0,
        left:0
    });
    canvas.add(group);


    const updateState = (newState)=>{
        state_title.set("text","STATE: "+newState);
    }


    return {
        onMsg: (msg) => {
                if(msg.data && msg.data.prop) {
                    switch(msg.data.prop) {
                        case "state":
                            updateState(msg.data.val);
                            break;
                    }
                }
        },
        setEnabled : (enabled) => {
            if(enabled) group.set("opacity",1);
            else group.set("opacity",0.35);
        }

    }
}
