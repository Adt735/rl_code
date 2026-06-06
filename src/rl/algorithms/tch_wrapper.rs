use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::{Device, nn};



#[derive(Serialize, Deserialize)]
#[serde(remote = "Device")]
pub enum DeviceDef {
    Cpu,
    Cuda(usize),
    Mps,
    Vulkan,
}


/// Wrapper around a `tch::nn`
pub struct NeuralNetwork {
    pub layers: Vec<i64>,
    pub end_with_softmax: bool,
    pub vs: nn::VarStore,
    pub model: nn::Sequential,
}
impl NeuralNetwork {
    pub fn new(device: Device, layers: Vec<i64>, end_with_softmax: bool) -> Self {
        assert!(layers.len() >= 2);

        let vs = nn::VarStore::new(device);

        let mut seq = nn::seq();

        for i in 0..layers.len() - 1 {
            let in_dim = layers[i];
            let out_dim = layers[i + 1];

            seq = seq.add(nn::linear(
                &vs.root() / format!("l{}", i),
                in_dim,
                out_dim,
                Default::default(),
            ));

            if i != layers.len() - 2 {
                seq = seq.add_fn(|x| x.relu());
            }
        }

        if end_with_softmax {
            seq = seq.add_fn(|x| x.softmax(-1, tch::Kind::Float))
        }

        Self {
            layers,
            end_with_softmax,
            vs,
            model: seq,
        }
    }

    pub fn forward(&self, x: &tch::Tensor) -> tch::Tensor {
        x.apply(&self.model)
    }

    pub fn varstore_size_bytes(&self) -> usize {
        self.vs.variables()
            .values()
            .map(|tensor| {
                tensor.numel() * tensor.kind().elt_size_in_bytes()
            })
            .sum()
    }

    pub fn save(&self, file_path: String) -> tch::Result<SerializableNetwork> {
        self.vs.save(&file_path)?;
        
        Ok(SerializableNetwork {
            layers: self.layers.clone(),
            end_with_softmax: self.end_with_softmax,
            file: file_path,
        })
    }

    pub fn load(
        device: Device,
        network: &SerializableNetwork
    ) -> tch::Result<Self> {
        let mut net = Self::new(device, network.layers.clone(), network.end_with_softmax);
        net.vs.load(&network.file)?;
        Ok(net)
    }
}


#[derive(Serialize, Deserialize)]
pub struct SerializableNetwork {
    pub layers: Vec<i64>,
    pub end_with_softmax: bool,
    pub file: String,
}


impl Serialize for NeuralNetwork {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Define a unique file name or convention for the weights
        let file_path = "network_weights.ot".to_string();
        
        self.vs.save(&file_path)
            .map_err(serde::ser::Error::custom)?;

        let proxy = SerializableNetwork {
            layers: self.layers.clone(),
            end_with_softmax: self.end_with_softmax,
            file: file_path,
        };

        proxy.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NeuralNetwork {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = SerializableNetwork::deserialize(deserializer)?;
        
        // Note: Defaulting to CPU here since device isn't stored in the proxy.
        // Change Device::Cpu to your preferred default or global configuration.
        let mut net = Self::new(Device::Cpu, proxy.layers, proxy.end_with_softmax);
        
        net.vs.load(&proxy.file).map_err(serde::de::Error::custom)?;

        Ok(net)
    }
}
