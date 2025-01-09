### Listing 1
import torch
import executorch.exir

# Die Klasse `Add´ definiert ein PyTorch-Programm für ein Modell,
# das die Tensoren `x´ und `y´ addiert.
# `Add´ ist eine Spezialisierung der  Klasse `torch.nn.Module´,
# mit der PyTorch ein neuronales Netzwerk definert.
class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()

# Die Methode `forward´ führt die Funktion des Netzwerks aus.
# Sie entspricht in komplexen Netzwerken der Inferenzfunktion.
  def forward(self, x: torch.Tensor, y: torch.Tensor):
      return x + y

pytorch_program = Add()

# Das API `export´ transformiert die Tensoroperationen in `Add´
# in eine Intermediate Representation (IR), ein internes Format,
# das PyTorch beim Transformieren von Callables (torch.nn.Module,
# Funktionen und Methoden) verwendet.
# Für die Codierung verwendet `export´ ATen dialect, das ein Subset
# (dialect) der IR für Operatoren der Tensor-Bibliothek (ATen) von
# PyTorch bezeichnet.
pytorch_graph = torch.export.export(pytorch_program, (torch.ones(1), torch.ones(1)))

# Das API `to_edge´ optimiert den Graphen von `Add´ in der Variablen
# `pytorch_graph´ zur Ausführung auf Edge-Devices und codiert
# das Ergebnis im Edge dialect, einem Subset der IR für Operatoren,
# die für mobile Endgeräte und Edge-Devices spezifisch sind.
edge_program = executorch.exir.to_edge(pytorch_graph)

# Das API `to_executorch´ transformiert den Graphen schließlich
# in Backend dialect, eine Variante von Edge dialect, die endgeräte-
# spezifische Operatoren ermöglicht. 
executorch_program = edge_program.to_executorch()

with open("add.pte", "wb") as file:
    file.write(executorch_program.buffer)
