# Reinforcement Learning Framework in Rust

Framework desenvolupat com a suport del Treball de Fi de Grau sobre comparació d'algorismes de Reinforcement Learning en entorns 2D discrets i continus.

## Característiques

### Algorismes implementats

Actualment el framework inclou implementacions dels següents algorismes:

* **Tabular Q-Learning**
* **Deep Q-Learning (DQN)**
* **Proximal Policy Optimization (PPO)**
* **Evolutionary Algorithms**

Tots els algorismes comparteixen una interfície comuna, fet que permet entrenar-los i comparar-los sobre qualsevol entorn compatible.

---

### Entorns disponibles

#### Grid 2D discret

Entorn senzill basat en una graella amb:

* Posició inicial de l'agent.
* Posició objectiu.
* Obstacles i parets.
* Diverses funcions de recompensa.

Ideal per a l'estudi de mètodes tabulars i comparacions inicials.

#### Entorn continu 2D

Entorn més complex amb:

* Moviment continu.
* Obstacles.
* Sensors tipus *ray-casting*.
* Simulació física mitjançant Rapier.
* Diferents estratègies de recompensa.

Especialment adequat per a algorismes basats en xarxes neuronals.

---

## Arquitectura del framework

El projecte està estructurat al voltant de dos conceptes principals:

### Entorns

Tots els entorns implementen:

```rust
EnvironmentTrait<S, A>
```

on:

* `S` és el tipus d'estat.
* `A` és el tipus d'acció.

Les funcions principals són:

```rust
fn get_state(&self) -> S;
fn step(&mut self, action: A) -> (f32, bool);
fn reset(&mut self);
```

---

### Algorismes

Tots els algorismes implementen:

```rust
RLAlgorithmTrait<S, A>
```

Les funcions mínimes requerides són:

```rust
fn train_epoch(&mut self, environment: &mut dyn EnvironmentTrait<S, A>, rng: &mut dyn RngCore);

fn best_action(&self, state: &S, actions: &[A]) -> Option<A>;
```

---

### Estats i accions

Els estats han d'implementar:

```rust
State
```

i les accions:

```rust
Action
```

Aquests traits proporcionen funcionalitats comunes, especialment la conversió a tensors per als algorismes basats en xarxes neuronals.

---

## Afegir un nou algorisme

Per implementar un nou algorisme cal:

### 1. Crear una nova estructura

Per exemple:

```rust
pub struct MyAlgorithm {
    ...
}
```

### 2. Implementar `RLAlgorithmTrait`

```rust
impl RLAlgorithmTrait<MyState, MyAction> for MyAlgorithm {
    ...
}
```

### 3. Afegir la configuració

Modificar:

```rust
src/rl/utils/cli.rs
```

i afegir una nova variant dins:

```rust
AlgoType
```

---

### 4. Afegir la càrrega de l'algorisme

Modificar la funció:

```rust
load_algo(...)
```

per permetre inicialitzar-lo des del fitxer TOML.

---

## Instal·lació

### Requisits

* Rust (última versió estable)
* Cargo
* FFmpeg (opcional, per generar vídeos)

### Instal·lar Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Comprovar:

```bash
rustc --version
cargo --version
```

### Instal·lar FFmpeg

#### Ubuntu

```bash
sudo apt install ffmpeg
```

#### Arch

```bash
sudo pacman -S ffmpeg
```

#### macOS

```bash
brew install ffmpeg
```

#### Windows

Instal·lar des de:

https://ffmpeg.org/download.html

i afegir-lo al `PATH`.

---

## Compilació

```bash
cargo build --release
```

---

## Ús

El framework funciona mitjançant fitxers de configuració TOML.

### Entorn discret

```bash
cargo run --release --bin discrete -- --config config.toml
```

### Entorn continu

```bash
cargo run --release --bin continuous -- --config config.toml
```

També es pot executar directament el binari:

```bash
./target/release/discrete --config config.toml
```

o

```bash
./target/release/continuous --config config.toml
```

---

## Exemple de configuració

```toml
algo = { ... }

env = { ... }

training = { ... }

action = { ... }
```

El fitxer de configuració defineix:

* L'algorisme a utilitzar.
* Els paràmetres de l'entorn.
* La configuració de l'entrenament.
* L'acció a executar (visualització, comparacions, estadístiques, etc.).

---

## Resultats generats

Segons la configuració seleccionada, el framework pot generar:

* Imatges de la trajectòria òptima.
* Gràfics d'entrenament.
* Vídeos de l'evolució de l'agent.
* Models entrenats en format JSON.

---

## Objectiu del projecte

Aquest framework s'ha desenvolupat amb finalitats acadèmiques per estudiar i comparar diferents paradigmes de Reinforcement Learning sobre problemes de navegació 2D, mantenint una arquitectura flexible que permet afegir fàcilment nous entorns i algorismes.
