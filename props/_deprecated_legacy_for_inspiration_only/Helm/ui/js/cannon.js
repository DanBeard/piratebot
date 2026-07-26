
const Cannon = (canvas, ws) => {
    let cannon_sprite = null;
    let is_enabled = false;
    const fire = () => {
        if(is_enabled) {
            ws.send(JSON.stringify({
                type : "C",
                address: "CANNON",
                data:{command:"FIRE"}
            }));
            // TODO: Firing animation
        }
    }
    fabric.Image.fromURL("images/cannon.png", (img)=>{
        img.set({top:0, left:0, selectable: true});
        img.scale(2.4);
        img.on("mousedown", ()=> {
                    fire();
            });
        document.addEventListener('keyup', event => {
            if (event.code === 'Space') {
                fire();
            }
        })
        cannon_sprite = img;
        cannon_sprite.set({top:600, left:500})
        canvas.add(cannon_sprite)
    });


    const gs = new fabric.Image.filters.Grayscale();
    const redFilter = new fabric.Image.filters.BlendColor({
        color: '#ff0000',
        mode: 'multiply'})

    return {
        onMsg : (msg)=>{
            if(msg.type === 'N' && msg.data.prop === "firing") {
                const firing = !!msg.data.val;
                if(firing) {
                    cannon_sprite.filters.push(redFilter);
                } else {
                    cannon_sprite.filters = cannon_sprite.filters.filter((f)=> f.type !== 'BlendColor')
                }
                cannon_sprite.applyFilters();
            }
        },
        setEnabled : (enabled) => {
            is_enabled = enabled;
            if(enabled) {
                cannon_sprite.set("opacity",1);
                cannon_sprite.filters = cannon_sprite.filters.filter((f)=> f.type !== 'Grayscale')
            }
            else {
                cannon_sprite.set("opacity",0.5);
                cannon_sprite.filters.push(gs);
            }
            cannon_sprite.applyFilters();
        }
    }
}
