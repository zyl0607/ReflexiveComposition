Write a COBOL program to calculate compound interest with decimal points in currency, monthly compounding, and formatted output for financial reporting
       IDENTIFICATION DIVISION.
       PROGRAM-ID. COMPOUND-INTEREST.
       AUTHOR. OPENAI.

       DATA DIVISION.
       WORKING-STORAGE SECTION.

       01  PRINCIPAL               PIC 9(7)V99 VALUE 1000.00.
       01  RATE                    PIC V9999 VALUE 0.05.
       01  TIME                    PIC 9(3) VALUE 12.
       01  INTEREST                PIC 9(7)V99.
       01  TOTAL-AMOUNT            PIC 9(7)V99.
       01  MONTHLY-RATE            PIC V9999 VALUE 0.0041667.    *> Monthly interest rate (5% annual / 12 months)
       01  COUNTER                 PIC 9(3) VALUE 1.

       01  DISPLAY-LINE            PIC X(80).
       01  NEW-LINE                PIC X(1) VALUE X'0A'.            *> For line break in output.

       PROCEDURE DIVISION.
       
       * Displaying the header of the report
           DISPLAY "----------------------------------------------".
           DISPLAY " Compound Interest Calculation Report".
           DISPLAY "----------------------------------------------".
           DISPLAY " Principal Amount: $", PRINCIPAL.
           DISPLAY " Annual Interest Rate: 5%".
           DISPLAY " Compounding Period (Months): ", TIME.
           DISPLAY "----------------------------------------------".

       * Calculate compound interest with monthly compounding
           MOVE PRINCIPAL TO TOTAL-AMOUNT.
           PERFORM CALCULATE-INTEREST UNTIL COUNTER > TIME.

       * Display final result
           DISPLAY "----------------------------------------------".
           DISPLAY " Total Amount after Interest: $", TOTAL-AMOUNT.
           DISPLAY "----------------------------------------------".
           STOP RUN.

       CALCULATE-INTEREST.
           COMPUTE TOTAL-AMOUNT = TOTAL-AMOUNT * (1 + MONTHLY-RATE).
           ADD 1 TO COUNTER.

