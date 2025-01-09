// ### Listing 2
// Speicherbereitstellung für Laufzeitumgebung ist Sache der App.
static uint8_t method_allocator_pool[4 * 1024U * 1024U];

int main(int argc, char** argv) {
  // Laufzeitumgebung initialisieren
  executorch::runtime::runtime_init();
  // PTE Datei mit ExecuTorch-Programm des Modells öffnen...
  Result<FileDataLoader> loader = FileDataLoader::from(argv[1]);
  // ... und in Laufzeitumgebung laden
  Result<Program> program = Program::load(&loader.get());
  
  // Metadaten der ersten Funktion im Modell laden
  const char* method_name = *program->get_method_name(0);
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  
  std::vector<Span<uint8_t>> planned_spans;
  // Speicherbereiche gemäß `method_meta´ in `planned_spans´ eintragen
  // (s. Code in examples/portable/executor_runner/executor_runner.cpp).
  // ...
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  MemoryManager memory_manager(&method_allocator, &planned_memory);
  
  // Funktion laden...
  Result<Method> method = program->load_method(method_name, &memory_manager);
  // ...Eingabedaten definieren...
  auto inputs = executorch::extension::prepare_input_tensors(*method);
  // ...Funktion ausführen...
  Error status = method->execute();
  // ...Ergebnisse holen und in `outputs´ speichern
  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
}
