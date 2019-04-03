const COLORS = ["lightcoral", "lightgreen", "lightblue", "wheat", "plum", "pink", "silver", "lightsalmon"];

const searchQuery = window.location.search.substr(1);
const DEFAULT_GRAPH = searchQuery.substr(0, 5) === "graph" ? Number(searchQuery.substr(6)) : 500;

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

function toScientific(number) {
    const exp = number.toExponential().split('e').map(String);
    return exp[0].slice(0, 5) + 'e' + exp[1];
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
        this.randomPlansEl = document.querySelector("#random_plans");
        this.randomPlansPercentEl = document.querySelector("#random_plans_percent");
        this.planButtonEl = document.querySelector("#plan");

        this.findBestEl = document.querySelector("#find_best");
        this.beamSizeEl = document.querySelector("#beam_size");

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
            this.scores = linearizations.map(l => toScientific(l.s) + " - " + l.r);
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
        this.planButtonEl.addEventListener("click", this.linearize.bind(this));
        window.addEventListener('resize', this.updateHeights.bind(this));


        graphsFetch.then(graphs => {
            this.graphs = graphs;

            const graphChange = e => {
                console.log(e);

                window.history.pushState(e.target.value, "Graph2Seq - " + e.target.value, "?graph=" + e.target.value);

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

    wrapLines(list) {
        const line = "<div class='line'>";
        return line + list.join("</div>" + line) + "</div>";
    }

    colorize(list, background = true) {
        console.debug("Colorize", list);
        list = list.map(s => s.toLowerCase());
        const nodes = Object.keys(this.colorMap);
        list.forEach((t, i) => {
            if (!nodes.every(n => t.indexOf(this.concatMap[n].toLowerCase()) > -1)) {
                list[i] = "<span class='strike'>" + t + "</span>";
            }
        });

        let text = this.wrapLines(list)

        nodes.forEach(n => {
            const concat = this.concatMap[n];
            const style = (background ? 'background' : 'border') + '-color: ' + this.colorMap[n]
            const rep = "<span class='entity' style='" + style + "'>" + n + "</span>";
            text = text.replaceAll(concat, rep)
        });

        if (background) {
            text = text.split(" ").map(w => {
                switch (w) {
                    case "[":
                        return "<span class='block'>";
                    case "]":
                        return "</span>";
                    default:
                        return w
                }
            }).join(" ").replaceAll("\\.", "<span class='separator'>.</span>");
        }

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

                console.log("Plans", res.linearizations.length);

                res.linearizations.forEach((l, i) => l["r"] = i + 1);

                // Filter plans randomly
                if (this.randomPlansEl.checked) {
                    const amount = Number(this.randomPlansPercentEl.value) - 1;
                    // res.linearizations = res.linearizations.slice(0, Math.max(amount, Math.floor(res.linearizations.length / 10)));


                    const step = Math.ceil(res.linearizations.length / amount);
                    console.log("amount", amount, "step", step);

                    const ids = [];
                    for (let i = 0; i < amount; i++) {
                        ids.push(step * i);
                    }

                    // const best_lin = res.linearizations[0];
                    // const ids = res.linearizations.map((_, i) => i).sort((a, b) => 0.5 - Math.random()).slice(0, this.randomPlansPercentEl.value - 1).sort((a, b) => a - b);
                    // if (ids[0] !== 0) {
                    //     ids.pop();
                    //     ids.unshift(0);
                    // }


                    ids.push(res.linearizations.length - 1);
                    console.log(ids);

                    res.linearizations = ids.map(i => res.linearizations[i]);
                }

                this.linearizations = res.linearizations;
                this.linearizationsEl.innerHTML = this.colorize(this.linearizations);

                this.scoresEl.innerHTML = this.wrapLines(this.scores);
                this.updateHeights();
            })
            .catch(e => {
                this.linearizationsEl.innerHTML = "Error";
                console.error(e)
            });
    }

    updateHeights() {
        window.requestAnimationFrame(() => {
            const lineHeights = Array.from(this.linearizationsEl.querySelectorAll("div.line")).map(l => l.clientHeight);

            [
                Array.from(this.scoresEl.querySelectorAll("div.line")),
                Array.from(this.sentencesEl.querySelectorAll("div.line"))
            ].forEach(lines => {
                lines.forEach((line, i) => {
                    line.style.lineHeight = lineHeights[i] + "px";
                });
            })
        });
    }

    translate() {
        this.sentencesEl.innerHTML = "Loading...";
        const body = {
            plans: this.linearizations,
            opts: {
                beam_size: Number(this.beamSizeEl.value),
                find_best: this.findBestEl.checked
            }
        };
        fetch("translate", {method: "POST", body: JSON.stringify(body)})
            .then(res => res.json())
            .then(res => {
                this.sentencesEl.innerHTML = this.colorize(res, false);
                this.updateHeights();

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