## Our Journey:
We set out to find a unique way to understand how market events impact public equities. We have experience using SEC filings in tandem with LLMs and stock signals to generate alpha, but for this hackathon we were drawn to finding an edge over the market using a unique alternative dataset. While considering X (formerly Twitter), we recalled accounts that tracked celebrities and CEOs using publicly available plane data. We hoped to replicate this result for corporate fleets of large, public healthcare, biotech, and pharmaceutical companies, and then try to generate insight about how their movement and location might signal a newsworthy event such as M&A. Accessing this data proved extremely challenging, as most corporations pay to remove their flight history from public sites. However, we obtained data for three corporate planes to prove our concept at a smaller scale. 

This acted as a proof of concept for a much more robust dataset. We paired flight data with information about relevant locations for each company and fed this into our AI model to generate daily confidence scores of significant company events occurring. 

One significant thing we encountered and learned from was the difficulty of access to alternative data. That said, we affirmed our belief that tangible insights can be produced with enough training data / time. As a next step, using alternative sources such as ADS-B, specific plane movement could be tracked without corporate entities being able to pay to hide their data. This was infeasible during the hackathon due to the large and unstructured nature of the ADS-B data: doing so would require downloading and parsing complete all worldwide flights, but could absolutely be accomplished with more time + storage, and our analysis indicates that doing so would yield valuable results.

## Our Product:
We designed an interactive, multi-modal dashboard to display corporate flight data, relevant stock pricing + history, and company news, as well as highlight our flight pattern significance score.

## ML Stack + Model Architecture:
![ML Stack](https://i.ibb.co/hFp7Nmdf/pipeline.png)
![Model](https://i.ibb.co/mrP9F7B5/model-architecture.png)

## Presentation:

[Slide Deck](https://docs.google.com/presentation/d/1QULZUTN9ClnbOcO46n7T5ai4f-bN8G3aanz7827rGvo/embed?start=false&loop=false&delayms=3000)

[Demo Video](https://www.youtube.com/watch?v=8a3PxUuf_gY)
