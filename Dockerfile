# Use Node.js 20 (required for Google GenAI SDK)
FROM node:20

# Create app directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy rest of the code
COPY . .

# Expose the port
EXPOSE 8080

# Start the server
CMD ["npm", "start"]
