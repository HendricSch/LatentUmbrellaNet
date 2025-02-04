## Wofür brauchen wir Wettervorhersagen?

- Naturkatastrophen frühzeitig erkennen

- Essentiell für viele Wirtschaften, z.B. Luft- und Raumfahrt, Aggragwirtschaft, Tourismus, uvm...

- Nützlich für tägliche Entscheidungen eines jeden Menschen

## Was meinen wir mit Wetter?

- Definition von Wetter: "Als Wetter (von althochdeutsch wetar „Wind, Wehen“) bezeichnet man den spürbaren, kurzfristigen Zustand der Atmosphäre (auch: messbarer Zustand der Troposphäre) an einem bestimmten Ort der Erdoberfläche, der unter anderem als Sonnenschein, Bewölkung, Regen, Wind, Hitze oder Kälte in Erscheinung tritt."

- D.h. wir betrachten eine Vielzahl von Atmosprährischen Variablen zu einem gewissen Zeitpunkt an einem festen Ort, z.B. Temperatur, Niederschlag, Windgeschwindigkeit, Luftfeuchtigkeit, ...

## Unsere Wunschvorstellung:

- Am liebsten hätte man ein "Wetterkarte", welche zu jedem Zeitpunkt und an jedem Ort auf der Erde die exakten Atmosphärischen Variablen modeliert.

- Selbst für die Gegenwart ist ein solches Modell nicht erstellbar! Warum?

- Wir haben nur die Möglichkeit an wenigen Stellen der Erde durch Messinstrumente, z.B. Wetterstationen, Satelitenaufnahmen, Wetterballons, Flugzeugen, Bojen, ... , bestimmte atmosphärische Variablen zu messen. Wie können nicht über die komplette Erdkugel an jeden qm dauerhaft die Temperatur messen.

## Die Lösung: Data Assimilation

- Nutzt verfügbare Messwerte und ein physikalisches Wettermodell, um ein kontinuierliche und lückenlose Aufnahme des aktuellen Wetters zu berechnen.

- Wichtig: Die Daten sind nich 100% akurat, da es sich nur um die möglichst sinnvolsten Daten handelt, welche das Modell berechnet.

## Reanalyse Daten

- https://www.dwd.de/DE/forschung/wettervorhersage/num_modellierung/02_datenassimilation/datenassimilation_node.html

## Unterschied zwischen NWP Modellen und ML Modellen:

- Hier stelle ich das IFS Modell von ECMRWF vor, dabei handelt es sich um das SOTA NWP Modell. Es basiert auf der physikalischen Modellierung des Wetters durch komplexe Differentialgleichungen und Modellierung phsikalischer Prozesse.

- Im Gegensatz dazu gibt es seit ca 2020 immer mehr ML Modelle, welche das Wetter komplett datengetrieben modellieren. Hier zeige ich zunächst die Anfänge durch die Arbeiten von Weyn et al. und gehe dann auf die beiden SOTA Modelle FourCastNet und GraphCast ein.

- Weyn et al.: Hier wird ein einfaches CNN Netzwerk mit zusätzlichen LSTM Layern genutzt, um Wetter Reanalse Daten der letzten Jahrzehnte zu lernen und vorherzusagen. Es wird sich auf die Nordhalbkugel beschränkt und eine geringe Auflösung verwendet. Es werden auch nur zwei Variablen modelliert: Die geopotentielle Höhe bei 500hpa und [noch einfügen]. Das Modell ist in der Lage einfache Baselines, wie z.b. die Persitenze oder Klimatologie zu schlagen, aber ist bei weitem noch nicht so gut, wie das IFS Modell.

- Das Modell wurde in einem Folgepaper weiterentwickelt und

## Welche Vorteile entstehen durch ML Modelle?

- Günstigere und sehr viel schnellere Berechnungen

- einfache und günstige Möglichkeit sehr viele Ensemple zu berechnen

- Mittlwereile sogar zum Teil genauere Berechnungen

## Was ist eine große Herausforderung bei den aktuellen ML Modellen?

Wetterdaten sind extrem groß! Google DeepMind sagt selbst, dass bei GraphCast wahrscheinlich die größten Datenoutputs in der Geschichte von ML erzeugt wurden.

### Die aktuellen Modelle versuchen auf verschiedene Arten mit dem Problem umzugehen:

- FourCastNet benutzt eine ViT Architektur, aber da die Daten so groß sind können sie keine Self-Attention Layer benutzten, da diese quadratisch skalieren, daher benutzen sie AFNOs für das Token-Mixing innerhalb der Transformer, welches nur eine linear-log Laufzeit hat.

- GraphCast benutzt eine Encoder-Processor-Decoder Architektur, welche die Rohdaten auf ein Graphen projeziert. Hierbei wird ein Graph mit 40.962 Nodes verwendet. Jeder Node speichert einen latenten Vektor der Dimension 474. lokal nahbeieinanderliegende Daten werden zusammen durch den Encoder Schritt auf den selben Node projeziert. 1440 x 720 x 227 = 235.353.600 -> 40.926 x 474 = 19.415.988 -> Kompression: 12,122

- Meine Idee: Wir arbeiten in einem durch einen pre-trained Autoencoder erstellten Latenten Raum, um eine hohe Datenkompression zu erhalten und die Daten handlicher für das eigentliche Forecast Modell zu machen. Dabei stammt die Idee von Stable Diffusion.

## Kurzer Ausflug zu Stable Diffusion:

- Diffusionsprozess wird genutzt um Bilder zu erstellen.

- Sehr rechenaufwändig, grade bei großen Bildern.
- Bei Latent Diffusion wird ein VAE benutzt, welcher auf allgemeine Bilder pre-trained wurde.

- Der Diffusionsprozess findet im Latenten Raum statt -> Training und Inferenz sind sehr viel schneller.

## LatentUmbrellaNet:

- Das Modell von meinem LatentUmbrellaNet vorstellen: Zunächst ein Übersichtsbild, auf welchem der ganze Ablauf dargestellt ist: Wetterdaten x0 -> Encoder -> latente Darstellung z0 -> ForeCastNet -> z1 -> ... -> zt -> Decoder -> xt

### Genauer auf den VQ-VAE eingehen:

- Was ist ein VQ-VAE? Auf die Architektur und den Quantisierungsschritt eingehen, wie funktioniert das Training?
- Was ist der Unterschied zu einem Autoencoder und Variational Autoencoder?
- Welchen Vorteil erhalten wird dadurch? -> Latente Darstellung besteht aus diskreten Codebook Vektoren -> Wir bekommen automatisch Tokens geliefert, welche wir für einen Transformeransatz nutzen können.

- Auf das aktuelle ForCastNet eingehen, also zeigen, dass es sich um eine einfache CNN Architektur handelt, welche in 6h Schritten die latenten Darstellungen des Wetters vorhersagt.

## Limitationen:

- Ich benutze zur Zeit eine niedrige Auflösung von den ERA5 Daten (128 x 64)

- Ich modelliere zur Zeit nur 5 Features auf jeweils nur einer Atmosphärenhöhe

- Durch diese Einschränkungen ist ein Vergleich mit aktuellen SOTA Modellen und IFS nicht sinnvoll, da die Auflösung und Anzahl der Variablen zu gering ist.

## Ergebnise:

- Grafik welche den RMSE zwischen meinen verschiedenen Modellen vergleicht. Das für alle 5 Variablen. Erkennt man irgendwelche Abweichungen bei den verschiedenen Variablen? Ich gehe mal davon aus, dass die Windgeschwindigkeit sehr scheiße sein wird, da es sich um ein sehr hochfrequentes Feature handelt, welches durch das VQ-VAE in der Form zerstört wird.

- Vergleich zwischen Orginal und Rekonstruierten Daten (VQ-VAE)

## Ausblick:

- Ich habe bis jetzt nur ein sehr simples CNN als ForeCastNet genommen. Ich möchte ausnutzen, dass meine latenten Daten als diskrete Tokens dargestellt werden und daher eine Transformer Architektur, wie beispielsweise ein GPT nutzen. Ich erhoffe mir dadurch sehr viel bessere Vorhersage, da Transformer Architekturen zur Zeit in nahezu jedem Bereich SOTA Lösungen erzielen. Außerdem wird ein VQ-GAN ein ähnlicher Ansatz verfolgt und das zeigt, dass die Ergebnise sehr zuversichtlich sein sollten.

- Ich benutze nur eine 2D Ansicht der Daten, da ich im Moment pro Feature nur ein Höhenlevel betrachte. Ich kann mir vorstellen, dass durch eine sinnvolles 3D-CNN der VQ-VAE so erweitert werden kann, dass dieser auch mehrere Höhenlevel pro Feature verarbeitet.

- Mein VQ-VAE neigt aktuell dazu einen sehr blury Output zu erzeugen. Ich benutze für das Training nur eine Reconstruction Loss und den Commitment Loss. Das Paper zu Tamming Transformer... hat gezeigt, dass man die Qualität der Rekonstruierten Daten extrem erhöhen kann, wenn man zusätzlich noch einen Adversial Loss und einen Perceptual Loss benutzt. Ich würde gerne einen PatchGAN Diskriminator nutzen, um einen Adversial Loss zu erzeugen und diesen in das Training meines VQ-VAE einfließen zu lassen. Ich erhoffe mir dadurch, dass mein Decoder in der Lage ist sehr viel genauerer Rekonstruktionen aus meinen latenten Darstellungen zu erzeugen. Für einen Perceptual Loss wird idR ein pre-trained Classification Network genutzt. Da es soetwas für Wetterdaten nicht gibt, wird es wohl nicht möglich sein einen Perceptual Loss zu integrieren.
