class TreeNode {
  constructor(value) {
    this.value = value;
    this.children = [];
  }
}

function convertTreeToVisData(root) {
  const nodes = [];
  const edges = [];

  function addNode(node) {
    if (!nodes.some(n => n.id === node.value.toString())) {
      nodes.push({ id: node.value.toString(), label: node.value.toString() });
    }
    node.children.forEach(child => {
      if (!nodes.some(n => n.id === child.value.toString())) {
        nodes.push({ id: child.value.toString(), label: child.value.toString() });
      }
      edges.push({ from: node.value.toString(), to: child.value.toString() });
      addNode(child);
    });
  }

  addNode(root);

  return { nodes, edges };
}

// Crea el árbol
function createTree(totalParentNodes, totalChildNodes, maxWidth, depth) {
  const maxNodes = totalParentNodes * totalChildNodes;
  if (maxNodes < Math.pow(maxWidth, depth-1)) {
    console.log("No se puede construir el árbol con los parámetros proporcionados.");
    console.log(`La profundidad máxima permitida es ${Math.log(maxNodes)/Math.log(maxWidth)+1}.`);
    return null;
  }

  const root = new TreeNode(1);
  const queue = [root];
  let currentNodeCount = 1;

  for (let i = 1; i < depth; i++) {
    let levelNodeCount = Math.min(currentNodeCount * totalChildNodes, maxNodes);
    let level = [];

    for (let j = 0; j < currentNodeCount; j++) {
      const parent = queue.shift();
      for (let k = 0; k < totalChildNodes && levelNodeCount > 0; k++, levelNodeCount--) {
        const child = new TreeNode(parent.value * totalChildNodes + k + 1);
        parent.children.push(child);
        level.push(child);
      }
    }

    queue.push(...level);
    currentNodeCount = level.length;
  }

  return root;
}


function guardarDatosArbol() {
  const totalParentNodes = document.getElementById("NumNodosPadre").value;
  const totalChildNodes = document.getElementById("NumNodosHijo").value;
  const maxWidth = document.getElementById("Amplitud").value;
  const depth = document.getElementById("Profundidad").value;

  // Crea el árbol utilizando los parámetros proporcionados
  const treeData = createTree(totalParentNodes, totalChildNodes, maxWidth, depth);

  // Si no se pudo crear el árbol, termina la función
  if (treeData === null) {
    return;
  }
  console.log(treeData)
  // Convierte el árbol en datos para visualización con vis.js
  const visData = convertTreeToVisData(treeData);

  // Configura la visualización con vis.js
  const container = document.getElementById("tree-container");
  const options = {
    layout: {
      hierarchical: {
        direction: "UD",
        sortMethod: "directed",
      },
    },
  };
  const network = new vis.Network(container, visData, options);
}
function guardarDatosRecorrido() {
  const nodoMeta = document.getElementById("NodoMeta").value;
  const busquedaPorProfundidad = document.getElementById("opcion1").checked;
  const busquedaPorAmplitud = document.getElementById("opcion2").checked;

  if (busquedaPorProfundidad) {
    console.log("Busqueda por profundidad");
  } else if (busquedaPorAmplitud) {
    console.log("Busqueda por amplitud");
  } else {
    console.log("Ninguna opción seleccionada");
  }

  // rest of your code here
}

