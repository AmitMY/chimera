const COLORS = ["lightcoral", "lightgreen", "lightblue", "wheat", "plum", "pink", "silver", "lightsalmon"];

const DEFAULT_GRAPH = 500;

String.prototype.replaceAll = function (search, replacement) {
    return this.replace(new RegExp(search, 'ig'), replacement);
};

function averagePrecision(vector) {
    let good = 0;
    let bad = 0;

    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
        if (vector[i]) {
            good += 1;
            sum += good / (good + bad)
        } else {
            bad += 1
        }
    }

    return good === 0 ? 0 : sum / good;
}

function successors(g, node) {
    const suc = g.successors(node);
    return [...new Set(suc.concat(suc.map(d => g.successors(d)).reduce((a, b) => a.concat(b), [])))]
}

const graphsFetch = fetch("graphs").then(res => res.json());

class Visualizer {
    constructor() {
        this.render = new dagreD3.render();
        this.graphs = {};

        this._linearizations = null;
        this.scores = null;
        this.concatMap = {};

        this.linearizationsEl = document.querySelector("#linearizations");
        this.scoresEl = document.querySelector("#scores");
        this.sentencesEl = document.querySelector("#output");
        this.precisionEl = document.querySelector("#precision");

        this.translateButtonEl = document.querySelector("#translate");
        this.graphSelectEl = document.querySelector("#graph-select");
        this.linearizeTypeEl = document.querySelector("#linearize_type");

        this.initZoom();
        this.initSync();
        this.initSettings();
    }

    get linearizations() {
        return this._linearizations;
    }

    set linearizations(linearizations) {
        console.debug("set linearizations", linearizations);

        if (Array.isArray(linearizations)) {
            this._linearizations = linearizations.map(l => l.l);
            this.scores = linearizations.map(l => l.s);
            this.translateButtonEl.removeAttribute("disabled");
        } else {
            this._linearizations = linearizations;
            this.translateButtonEl.setAttribute("disabled", "");
        }
    }

    initZoom() {
        const svg = d3.select("svg"),
            inner = d3.select("svg g"),
            zoom = d3.zoom().on("zoom", () => inner.attr("transform", d3.event.transform));
        svg.call(zoom);
    }

    initSync() {
        const elements = [this.linearizationsEl, this.scoresEl, this.sentencesEl];

        let lastTarget;
        const sync = (e) => {
            if (!lastTarget) {
                window.requestAnimationFrame(() => {
                    const scrollTop = lastTarget.scrollTop;
                    elements.filter(el => el !== e.target).forEach(el => el.scrollTop = scrollTop);
                    lastTarget = null;
                });
            }
            lastTarget = e.target;
        };
        elements.forEach(el => el.addEventListener("scroll", sync));
    }

    initSettings() {
        this.translateButtonEl.addEventListener("click", this.translate.bind(this));
        this.linearizeTypeEl.addEventListener("change", this.linearize.bind(this));
        graphsFetch.then(graphs => {
            this.graphs = graphs;

            const graphChange = e => {
                console.log(e);
                this.graph = graphs[Number(e.target.value)];

                this.colorMap = {};
                Array.from(new Set(this.graph.map(([s, l, o]) => [s, o]).reduce((a, b) => a.concat(b))))
                    .forEach((n, i) => this.colorMap[n] = COLORS[i]);

                this.draw();
            };
            graphChange({target: {value: DEFAULT_GRAPH}});

            this.graphSelectEl.addEventListener('change', graphChange);

            window.requestAnimationFrame(() => {
                graphs.forEach((g, i) => {
                    const option = document.createElement("option");
                    option.value = i;
                    option.innerText = i + " - " + g.length;
                    if (i === DEFAULT_GRAPH) {
                        option.selected = "selected";
                    }

                    this.graphSelectEl.appendChild(option);
                });
            });

            const graphSelectorContainer = document.querySelector("#graphs-selector");
            new Set(graphs.map(g => g.length)).forEach(size => {
                const b = document.createElement("button");
                b.innerText = size;
                b.addEventListener("click", () => {
                    const relevantGraphs = graphs.map((g, i) => [g, i]).filter(([g, i]) => g.length === size);
                    const randomGraph = relevantGraphs[Math.floor(Math.random() * relevantGraphs.length)];

                    this.graphSelectEl.selectedIndex = randomGraph[1];
                    graphChange({target: {value: randomGraph[1]}});
                });
                graphSelectorContainer.appendChild(b);
            });

            function rollingDownload() {
                let p = Promise.resolve();
                Object.keys(graphs).forEach(k => {
                    if (k.substr(0, 4) === "test") {

                        p = p.then(() => {
                            graphChange({target: {value: k}});
                            return exportImage(k.split(" - ")[1])
                        });
                    }
                })
            }

            console.log(rollingDownload)
        });
    }

    draw() {
        const g = new dagreD3.graphlib.Graph({compound: true}).setGraph({})
            .setDefaultEdgeLabel(() => ({}));

        const nodes = Object.keys(this.colorMap);
        nodes.forEach((n, i) => g.setNode(i, {label: n, style: "fill: " + this.colorMap[n]}));
        this.graph.forEach(([s, label, o]) => g.setEdge(nodes.indexOf(s), nodes.indexOf(o), {label}));


        // Set margins
        g.marginx = 20;
        g.marginy = 20;

        // Round corners
        const nodesSuc = g.nodes().map(v => [v, successors(g, v).length]).sort((a, b) => a[1] - b[1]);
        nodesSuc.forEach(([n, _]) => {
            const groupName = n + "g";
            // g.setNode(groupName, {label: groupName, style: 'fill: #d3d7e8;stroke:black'});
            // successors(g, n).concat([n]).forEach(d => g.setParent(d, groupName));

            // const pred = g.predecessors(n);
            // pred.forEach((p) => {
            //     g.removeEdge(p, n);
            //     g.setEdge(p, groupName, {label: "a"})
            // })
        });

        g.nodes().forEach((v) => {
            const node = g.node(v);
            node.rx = node.ry = 5;
        });

        g.transition = (selection) => selection.transition().duration(500);

        // Render the graph into svg g
        const svg = d3.select("svg");
        const svgGroup = svg.select("g");
        this.render(svgGroup, g);

        // svg.selectAll("g.edgeLabel").on("click", ({v, w}) => {
        //     const edge = g.edge(v, w);
        //
        //     const label = (edge.charAt(0) === ">" ? "<": ">") +edge.label.substr(1, edge.label.length - 1);
        //     const graphHashs = this.graph.map(([a, b, c]) => a + b + c);
        //     const currentEdge = nodes[v] + edge.label + nodes[w];
        //
        //     this.graph.splice(graphHashs.indexOf(currentEdge), 1);
        //     this.graph.push([nodes[w], label, nodes[v]]);
        //
        //     this.draw();
        // });

        this.linearize();


        // Center the graph
        const width = svg._groups[0][0].getBoundingClientRect().width;
        const xCenterOffset = (width - g.graph().width) / 2;
        svgGroup.attr("transform", "translate(" + xCenterOffset + ", 20)");
        svg.attr("height", g.graph().height + 40);
    }

    colorize(list) {
        console.debug("Colorize", list);
        list = list.map(s => s.toLowerCase());
        const nodes = Object.keys(this.colorMap);
        list.forEach((t, i) => {
            if (!nodes.every(n => t.indexOf(this.concatMap[n].toLowerCase()) > -1)) {
                list[i] = "<span class='strike'>" + t + "</span>";
            }
        });
        let text = list.join("<br />");
        nodes.forEach(n => {
            const concat = this.concatMap[n];
            const rep = "<span class='entity' style='background-color: " + this.colorMap[n] + "'>" + n + "</span>";
            text = text.replaceAll(concat, rep)
        });
        return text.replaceAll("_", " ");
    }

    linearize() {
        this.sentencesEl.innerHTML = this.scoresEl.innerHTML = this.precisionEl.innerHTML = "";
        this.linearizationsEl.innerHTML = "Loading...";
        this.linearizations = null;
        const linearizeType = this.linearizeTypeEl.checked ? 'full' : 'partial';
        fetch("plans/" + linearizeType, {method: "POST", body: JSON.stringify(this.graph)})
            .then(res => res.json())
            .then(res => {
                this.concatMap = res.concat;
                this.linearizations = res.linearizations;
                this.linearizationsEl.innerHTML = this.colorize(this.linearizations);
                // this.scoresEl.innerHTML = this.scores.map(String).map(x => x.substr(0, 10)).join("<br />")
            })
            .catch(e => {
                this.linearizationsEl.innerHTML = "Error";
                console.error(e)
            });
    }

    translate() {
        this.sentencesEl.innerHTML = "Loading...";
        fetch("translate", {method: "POST", body: JSON.stringify(this.linearizations)})
            .then(res => res.json())
            .then(res => {
                this.sentencesEl.innerHTML = this.colorize(res);

                const relVec = this.sentencesEl.innerHTML.split("<br>").map(x => x.indexOf('"strike"') === -1);
                this.precisionEl.innerHTML = averagePrecision(relVec);

                for (let i = 0; i < 5; i++) {
                    console.log("Shuffle Precision", averagePrecision(relVec.sort(() => .5 - Math.random())))
                }
            })
            .catch(e => {
                this.sentencesEl.innerHTML = "Error";
                console.error(e)
            });
    }
}

document.addEventListener('DOMContentLoaded', () => new Visualizer());


function testImage(url, timeoutT) {
    return new Promise(function (resolve, reject) {
        var timeout = timeoutT || 5000;
        var timer, img = new Image();
        img.onerror = img.onabort = function () {
            clearTimeout(timer);
            reject("error");
        };
        img.onload = function () {
            clearTimeout(timer);
            resolve("success");
        };
        timer = setTimeout(function () {
            // reset .src to invalid URL so it stops previous
            // loading, but doesn't trigger new load
            img.src = "//!!!!/test.jpg";
            reject("timeout");
        }, timeout);
        img.src = url;
    });
}

function exportImage(name) {
    const canvas = document.getElementById("canvas");
    const svg = document.querySelector("svg");
    Array.from(svg.querySelectorAll(".edgePath .path")).forEach(p => {
        p.setAttribute("stroke", "black");
        p.setAttribute("stroke-width", "2")
    });

    return new Promise(res => {
        canvg(canvas, svg.outerHTML, {
            renderCallback: () => {
                const link = document.createElement('a');
                document.body.appendChild(link);

                const i = setInterval(() => {
                    requestAnimationFrame(() => {
                        const canvasData = canvas.toDataURL("image/png");
                        testImage(canvasData)
                        if (canvasData !== "data:,") {
                            console.log(canvasData)
                            clearInterval(i);

                            setTimeout(() => {
                                link.setAttribute('download', name + ".png");
                                link.setAttribute('href', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
                                link.click();
                                document.body.removeChild(link);
                                res();
                            }, 100)
                        }
                    });
                }, 100);

            }
        })
    })
}