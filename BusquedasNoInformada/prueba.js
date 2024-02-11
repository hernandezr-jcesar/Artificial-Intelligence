let recorrido = [];
let data;
let network;

function generarArbol() {
  const numNodosPadre = parseInt(document.getElementById('NumNodosPadre').value);
  const numNodosHijo = parseInt(document.getElementById('NumNodosHijo').value);
  const amplitud = parseInt(document.getElementById('Amplitud').value);
  const profundidad = parseInt(document.getElementById('Profundidad').value);

  data = {
    nodes: [],
    edges: []
  };

  // Agregar nodo raíz
  data.nodes.push({ id: 1, label: 'Nodo 1', level: 0 }); // Agregar nivel (level) para orden jerárquico

  let nodeId = 2; // ID de nodo actual

  // Función para construir el árbol recursivamente
  function buildTree(parentNodeId, depth, level) {
    if (depth >= profundidad) return;

    for (let i = 0; i < amplitud; i++) {
      const childNodeId = nodeId++;
      data.nodes.push({ id: childNodeId, label: `Nodo ${childNodeId}`, level: level + 1 }); // Agregar nivel (level) para orden jerárquico
      data.edges.push({ from: parentNodeId, to: childNodeId });
      buildTree(childNodeId, depth + 1, level + 1); // Incrementar el nivel en la llamada recursiva
    }
  }

  // Construir el árbol comenzando desde el nodo raíz
  buildTree(1, 0, 0); // El nivel de la raíz es 0

  const container = document.getElementById('tree-container');
  const options = {
    layout: {
      hierarchical: {
        direction: 'UD', // Cambiar la dirección del árbol hacia abajo (Up-Down)
        sortMethod: 'directed' // Ordenar los nodos en función de las conexiones
      }
    }
  };
  network = new vis.Network(container, data, options);
}

function guardarDatosRecorrido() {
  const nodoMeta = parseInt(document.getElementById('NodoMeta').value);
  const opcionProfundidad = document.getElementById('opcion1').checked;
  const opcionAmplitud = document.getElementById('opcion2').checked;

  recorrido = [];

  if (opcionProfundidad) {
    recorridoPorProfundidad(1, nodoMeta);
  } else if (opcionAmplitud) {
    recorridoPorAmplitud(1, nodoMeta);
  }

  mostrarRecorrido();
}

function recorridoPorProfundidad(startNodeId, nodoMeta) {
    const stack = [startNodeId];
    const visited = new Set();
  
    while (stack.length > 0) {
      const nodeId = stack.pop();
      visited.add(nodeId);
      recorrido.push(nodeId);
  
      if (nodeId === nodoMeta) {
        return;
      }
  
      const neighbors = network.getConnectedNodes(nodeId);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          stack.push(neighbor);
          visited.add(neighbor);
        }
      }
    }
  }
  

function recorridoPorAmplitud(startNodeId, nodoMeta) {
  const queue = [startNodeId];
  const visited = new Set();

  while (queue.length > 0) {
    const nodeId = queue.shift();
    visited.add(nodeId);
    recorrido.push(nodeId);

    if (nodeId === nodoMeta) {
      return;
    }

    const neighbors = network.getConnectedNodes(nodeId);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        queue.push(neighbor);
        visited.add(neighbor);
      }
    }
  }
}

function mostrarRecorrido() {
  const resultadoDiv = document.getElementById('resultado');
  resultadoDiv.innerHTML = '';

  if (recorrido.length === 0) {
    resultadoDiv.textContent = 'No se encontró el nodo meta.';
  } else {
    const recorridoStr = recorrido.join(' → ');
    resultadoDiv.textContent = `Recorrido: ${recorridoStr}`;
  }
}
