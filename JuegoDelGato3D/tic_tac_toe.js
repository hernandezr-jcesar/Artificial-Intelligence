    const planes = document.querySelectorAll('.plane');
    const cells = document.querySelectorAll('.cell');
    const winningCombinations = [
      // Rows
      [0, 1, 2], [3, 4, 5], [6, 7, 8],
      [9, 10, 11], [12, 13, 14], [15, 16, 17],
      [18, 19, 20], [21, 22, 23], [24, 25, 26],
      // Columns
      [0, 9, 18], [1, 10, 19], [2, 11, 20],
      [3, 12, 21], [4, 13, 22], [5, 14, 23],
      [6, 15, 24], [7, 16, 25], [8, 17, 26],
      // Diagonals
      [0, 10, 20], [2, 10, 18],
      [6, 12, 18], [8, 12, 20],
      [0, 9, 18], [2, 11, 20],
      [6, 15, 24], [8, 17, 26],
      [0, 3, 6], [1, 4, 7], [2, 5, 8],
      [9, 12, 15], [10, 13, 16], [11, 14, 17],
      [18, 21, 24], [19, 22, 25], [20, 23, 26],
      [0, 4, 8], [2, 4, 6],
      [9, 13, 17], [11, 13, 15],
      [18, 22, 26], [20, 22, 24]
    ];
    let currentPlayer = 'X';
    let gameEnded = false;
    let winner = null;
    
    // Obtén el elemento del mensaje de turno
    const turnMessage = document.getElementById("turn-message");

    cells.forEach(cell => {
      cell.addEventListener('click', handleCellClick);
      turnMessage.textContent = `Turno del jugador ${currentPlayer}`;
    });

    function handleCellClick(event) {
      const cell = event.target;
      
      if (cell.textContent || gameEnded) return;
      
      cell.textContent = currentPlayer;
      cell.style.backgroundColor = currentPlayer === 'X' ? '#ff9a8d' : '#a0c4ff';
      
      if (checkWin()) {
        gameEnded = true;
        setTimeout(() => {
          alert(`Player ${currentPlayer} wins!`);
          resetGame();
        }, 200);
      } else if (checkDraw()) {
        gameEnded = true;
        setTimeout(() => {
          alert("It's a draw!");
          resetGame();
        }, 200);
      } else {
        currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
      }
      changeTurn();
    }

    function checkWin() {
      const cellsArray = Array.from(cells);
      const winner = winningCombinations.find(combination => {
        const [a, b, c] = combination;
        return (
          cellsArray[a].textContent === currentPlayer &&
          cellsArray[b].textContent === currentPlayer &&
          cellsArray[c].textContent === currentPlayer
        );
      });

      if (winner) {
        cellsArray.forEach((cell, index) => {
          if (winner.includes(index)) {
            cell.classList.add('winning-cell');
          }
        });
        turnMessage.textContent = `Player ${currentPlayer} wins!`;
        return true;
      }
    
      return false;
    }

    function checkDraw() {
      return Array.from(cells).every(cell => cell.textContent);
    }

    function resetGame() {
      // Recargar la página
      location.reload();
    }

      // Función para cambiar el turno
    function changeTurn() {
      turnMessage.textContent = `Turno del jugador ${currentPlayer}`;
    }
    
