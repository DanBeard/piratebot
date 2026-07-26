
const PirateShip = (canvas, ws, ship_sprite) => {

    ship_sprite.set({top:500, left:500})
    canvas.add(ship_sprite);

    const gs = new fabric.Image.filters.Grayscale();

    return {
        onMsg : (msg)=>{

        },
        setEnabled : (enabled) => {
            if(enabled) {
                ship_sprite.set("opacity",1);
                ship_sprite.filters.shift();
            }
            else {
                ship_sprite.set("opacity",0.5);
                ship_sprite.filters.push(gs);
            }
            ship_sprite.applyFilters();
        }
    }
}
