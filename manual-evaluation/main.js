document.addEventListener('DOMContentLoaded', async () => {
    const samples = await (await fetch("samples.json")).json();
    console.log(samples)

    const downloadLink = document.createElement("a")
    downloadLink.setAttribute("download", "samples.json");
    document.body.appendChild(downloadLink);

    const downloadButton = document.createElement("button");
    downloadButton.innerText = "DOWNLOAD";
    downloadButton.addEventListener("click", () => {
        downloadLink.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(samples));
        downloadLink.click()
    });
    document.body.appendChild(downloadButton)

    samples.forEach(({id, sen, rdf, hal}, i) => {
        const el = document.createElement("div");

        const title = document.createElement("h3");
        title.innerHTML = id + ". " + sen;
        el.appendChild(title);

        document.body.appendChild(el);

        const table = document.createElement("table");
        el.appendChild(table)
        const header = document.createElement("tr");
        header.innerHTML = "<td>S</td><td>R</td><td>O</td><td>Exists</td><td>Doesn't exist</td><td>Wrong Lexicalization</td><td>Wrong REG</td>";
        table.appendChild(header);

        rdf.forEach(([s, r, o, res], j) => {
            const row = document.createElement("tr");
            row.innerHTML = "<td>" + s + "</td><td>" + r + "</td><td>" + o + "</td>";
            table.appendChild(row);

            ["yes", "no", "no-lex", "no-reg"].forEach((val) => {
                const radio = document.createElement("input");
                radio.type = "radio";
                radio.name = i + "_" + j;
                if (res === val) {
                    radio.checked = true;
                }
                radio.addEventListener("change", e => {
                    samples[i].rdf[j][3] = val;
                });
                const td = document.createElement("td")
                td.appendChild(radio);
                row.appendChild(td)
            });
        });

        const input = document.createElement("input");
        input.value = hal === null ? 0 : hal;
        input.addEventListener("change", (e) => {
            let val = e.target.value;
            if (isNaN(val)) {
                val = null;
                e.target.value = 0
            }
            samples[i].hal = Number(val);
        });
        el.appendChild(input)
    })
});
