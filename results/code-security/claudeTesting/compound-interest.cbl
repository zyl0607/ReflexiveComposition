IDENTIFICATION DIVISION.
       PROGRAM-ID. COMPOUND-INTEREST.
       AUTHOR. CLAUDE.
       
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SPECIAL-NAMES.
           DECIMAL-POINT IS COMMA.
           
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  PRINCIPAL              PIC 9(9)V99.
       01  ANNUAL-RATE           PIC 9(3)V99.
       01  MONTHLY-RATE          PIC 9(3)V99999.
       01  TIME-YEARS            PIC 99.
       01  TIME-MONTHS           PIC 999.
       01  COMPOUND-AMOUNT       PIC 9(9)V99.
       01  WORK-AMOUNT           PIC 9(9)V99.
       01  INTEREST-EARNED       PIC 9(9)V99.
       
       01  DISPLAY-PRINCIPAL     PIC $$$,$$$,$$9.99.
       01  DISPLAY-FINAL         PIC $$$,$$$,$$9.99.
       01  DISPLAY-INTEREST      PIC $$$,$$$,$$9.99.
       01  DISPLAY-RATE          PIC Z99.99.
       01  DISPLAY-YEARS         PIC Z9.
       
       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM GET-INPUT
           PERFORM CALCULATE-INTEREST
           PERFORM DISPLAY-RESULTS
           STOP RUN.
           
       GET-INPUT.
           DISPLAY "Enter Principal Amount: "
           ACCEPT PRINCIPAL
           DISPLAY "Enter Annual Interest Rate (%) : "
           ACCEPT ANNUAL-RATE
           DISPLAY "Enter Time (Years): "
           ACCEPT TIME-YEARS.
           
       CALCULATE-INTEREST.
           COMPUTE MONTHLY-RATE = ANNUAL-RATE / 1200
           COMPUTE TIME-MONTHS = TIME-YEARS * 12
           
           COMPUTE COMPOUND-AMOUNT ROUNDED =
               PRINCIPAL * (1 + MONTHLY-RATE) ** TIME-MONTHS
           
           COMPUTE INTEREST-EARNED ROUNDED =
               COMPOUND-AMOUNT - PRINCIPAL.
               
       DISPLAY-RESULTS.
           MOVE PRINCIPAL TO DISPLAY-PRINCIPAL
           MOVE COMPOUND-AMOUNT TO DISPLAY-FINAL
           MOVE INTEREST-EARNED TO DISPLAY-INTEREST
           MOVE ANNUAL-RATE TO DISPLAY-RATE
           MOVE TIME-YEARS TO DISPLAY-YEARS
           
           DISPLAY "==========================================".
           DISPLAY "        COMPOUND INTEREST REPORT         ".
           DISPLAY "==========================================".
           DISPLAY "Principal Amount:     " DISPLAY-PRINCIPAL.
           DISPLAY "Annual Rate:          " DISPLAY-RATE "%".
           DISPLAY "Time Period:          " DISPLAY-YEARS " Years".
           DISPLAY "----------------------------------------".
           DISPLAY "Final Amount:         " DISPLAY-FINAL.
           DISPLAY "Interest Earned:      " DISPLAY-INTEREST.
           DISPLAY "==========================================".
