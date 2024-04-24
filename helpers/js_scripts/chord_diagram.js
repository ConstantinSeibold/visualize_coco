const width = 640;
const height = width;

var svg = d3.select("div").append("svg")
 .attr("id", "canvas")    
 .attr("width", width)
 .attr("height", height)
 ;

data = [
    1,2,3
    // {"id":1},
    // {"id":2},
    // {"id":3},
    // {"id":4},
    // {"id":5},
    ];
a
// Object.assign([
//     [11975,  5871, 8916, 2868],
//     [ 1951, 10048, 2060, 6171],
//     [ 8010, 16145, 8090, 8045],
//     [ 1013,   990,  940, 6907]
//     ], {
//     names: ["black", "blond", "brown", "red"],
//     colors: ["#000000", "#ffdd89", "#957244", "#f26223"]
// })

// svg
//     .append("circle")
//     .attr("r", 20)
//     .attr("cx","50%")
//     .attr("cy","50%")
//     .style("fill","green");

svg.selectAll("p")
    .data(data)
    .enter()
    .append("p")
    .text(dta => dta.id)
    ;

    